"""
Three ships, one sea: race the fast / balanced / eco agents across a changing wind map.

Built for an audience that has never heard of trajectory optimisation. The three agents are
identical networks trained on the same wind, differing only in what they were told to value:
arriving early, or arriving cheaply. They leave the same port at the same moment, sail the
same weather, and the animation lets you watch the disagreement play out - the impatient ship
pulls ahead through the gale, the thrifty one hangs back and lets the wind do the work.

Nothing on screen is jargon: the readouts are hours at sea and fuel burned, the fuel bars are
scaled to the thirstiest ship so the punchline ("same trip, a quarter of the fuel") is readable
without a caption.

    python evolving_race.py                                    # the default demo
    python evolving_race.py --field-seed 26 --start 0.6 5.0 --goal 9.4 5.4 --tag _east
    python evolving_race.py --dpi 100 --stride 1 --map-every 1 # slower, smoother, ~40 MB

Start and destination sit at opposite corners, 10.8 units apart on a 12-unit map, so the voyage
is long enough for the three priorities to separate visibly.

CHOOSING A CASE. These clones arrive on roughly a third of crossings (see the multi-objective
section of the README), so a demo case has to be picked, not assumed: `--field-seed` and the
route were selected by sweeping 40 generated fields x 3 routes and keeping the cases where all
three ships arrive, in the expected order, by visibly different paths. Seeds 31 (the default,
corner to corner) and 26 (due east) both hold up at the renderer's own weather fidelity; a case
that looks fine at fewer `--slices` can change outcome at 49, so re-check after changing them.

Outputs: output/race_<scenario><tag>.gif and output/race_<scenario><tag>_panels.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter

from env import ShipEnv
from wind import WindField, generate_wind_field
from benchmark_dp import load_model
from wind_obs import wrap_wind_obs
from receding_horizon_demo import EvolvingWind, simulate
from evolving_scenario_demo import BUILDERS
from viz import windy_cmap, windy_norm, wind_scale_ticks
import preferences as P

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

INK = "#04121f"
GLOW = [pe.withStroke(linewidth=4.5, foreground=INK, alpha=0.9)]
TEXT_GLOW = [pe.withStroke(linewidth=3.0, foreground=INK, alpha=0.95)]

# Bright track colours, chosen to sit outside the blue-teal-green middle of the wind palette.
# (preferences.COLORS is the print palette; on a dark animated map it disappears.)
NEON = {"fast": "#ff5714", "balanced": "#ffe74c", "eco": "#35f0c0"}

# what each agent is called in front of an audience
PLAIN = {
    "fast": "IN A HURRY",
    "balanced": "BALANCED",
    "eco": "FUEL SAVER",
}
PLAIN_SUB = {
    "fast": "get there first, fuel is cheap",
    "balanced": "a sensible compromise",
    "eco": "burn as little as possible",
}


_BIG_FIELD = {}


def drifting_weather(frac, seed=5, drift=(4.5, 2.0), extent=(-1.0, 11.0), pad=6.0, dx=0.12):
    """
    A generated wind field sliding across the domain: the same weather the agents were
    trained on, in motion.

    The synthetic set pieces in `evolving_scenario_demo.py` (cyclones at strength 11, gale
    fronts) are deliberately stronger and more structured than anything the wind generator
    produces, which is what makes them good stress tests and bad demos - the clones are far
    outside their training distribution there and simply thrash. Here the field is generated
    by the ordinary generator on an oversized domain, and the visible window is a crop that
    translates with time. Magnitudes and correlation lengths are exactly those of training;
    only the position changes, so the weather moves without ever leaving the distribution.
    """
    key = (seed, extent, pad, dx)
    if key not in _BIG_FIELD:
        lo, hi = extent[0] - pad, extent[1] + pad
        n = int(round((hi - lo) / dx)) + 1
        _BIG_FIELD[key] = generate_wind_field(seed, nx=n, ny=n, extent=(lo, hi))
    big = _BIG_FIELD[key]
    x = np.linspace(extent[0], extent[1], 101)
    y = np.linspace(extent[0], extent[1], 111)
    X, Y = np.meshgrid(x, y, indexing="ij")
    wx, wy = big(X + drift[0] * frac, Y + drift[1] * frac)
    return WindField(x, y, wx, wy)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefs", nargs="+", default=P.ORDER, choices=list(P.PREFERENCES))
    ap.add_argument("--tag-prefix", default="pref", help="agent <p> is models/<tag-prefix>_<p>.zip")
    ap.add_argument("--models", nargs="*", default=None, metavar="PREF=PATH")
    ap.add_argument("--scenario", default="drift", choices=["drift"] + sorted(BUILDERS),
                    help="'drift': a generated field sliding across the domain (in the agents' "
                         "training distribution, the one to demo with). The others are the "
                         "synthetic set pieces of evolving_scenario_demo.py - far stronger than "
                         "anything the generator makes, so the clones thrash in them.")
    ap.add_argument("--field-seed", type=int, default=31, help="'drift': which generated field")
    ap.add_argument("--drift", type=float, nargs=2, default=[6.0, 3.0],
                    help="'drift': how far the weather slides over the voyage")
    ap.add_argument("--start", type=float, nargs=2, default=[0.8, 1.4])
    ap.add_argument("--goal", type=float, nargs=2, default=[9.0, 8.4])
    ap.add_argument("--slices", type=int, default=49, help="map snapshots across the window")
    ap.add_argument("--window-h", type=float, default=24.0, help="real hours the window spans")
    ap.add_argument("--update-every-min", type=float, default=90.0,
                    help="how often each agent is handed a refreshed map")
    ap.add_argument("--ms-per-unit", type=float, default=2.5)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--hold-s", type=float, default=1.6, help="seconds to hold the final frame")
    ap.add_argument("--dpi", type=int, default=76, help="animation resolution; the GIF is a full-frame "
                                                        "animated map, so this drives the file size")
    ap.add_argument("--gif-colors", type=int, default=0,
                    help="requantise the finished GIF onto one shared palette. Off by default: it "
                         "discards the per-frame cropping Pillow already did, and here that costs "
                         "more than the palette saves.")
    ap.add_argument("--map-every", type=int, default=4,
                    help="repaint the wind map every N frames (the ships still move every frame). "
                         "A background that is identical between frames is what lets the GIF store "
                         "only the changed region, so this is the main lever on file size.")
    ap.add_argument("--tag", default="")
    return ap.parse_args()


def agent_policy(model, obs_cfg, host, wrapper):
    """A policy(state, field) -> thrust closure over one SB3 wind-aware model."""
    def policy(state, field):
        host.wind = field
        host.state = np.asarray(state, dtype=np.float64)
        a, _ = model.predict(wrapper.observation(None), deterministic=True)
        return a
    return policy


def fuel_curve(actions, p):
    """Cumulative control energy after each step, prepended with 0 (so it aligns with traj)."""
    if len(actions) == 0:
        return np.zeros(1)
    return np.concatenate(([0.0], np.cumsum((np.asarray(actions) ** 2).sum(axis=1) * p.dt)))


def shrink_gif(path, colors=128):
    """
    Requantise a finished GIF onto ONE shared palette.

    Every frame here is a full repaint of a photographic wind map, so GIF's inter-frame
    optimisation has almost nothing to work with and Pillow's per-frame palettes make it
    worse (each frame carries its own colour table, and the background shimmers as the
    palettes disagree). One adaptive palette taken from the middle of the voyage fixes both.
    """
    from PIL import Image, ImageSequence
    im = Image.open(path)
    duration = im.info.get("duration", 50)
    frames = [f.copy().convert("RGB") for f in ImageSequence.Iterator(im)]
    im.close()
    pal = frames[len(frames) // 2].quantize(colors=colors)
    out = [f.quantize(palette=pal) for f in frames]
    out[0].save(path, save_all=True, append_images=out[1:], duration=duration, loop=0,
                optimize=True, disposal=1)
    return os.path.getsize(path)


def _paint(ax, field, ms_per_unit, step=5):
    im = ax.pcolormesh(field.x, field.y, field.speed.T, shading="gouraud",
                       cmap=windy_cmap(), norm=windy_norm(ms_per_unit))
    X, Y = np.meshgrid(field.x, field.y, indexing="ij")
    q = ax.quiver(X[::step, ::step], Y[::step, ::step], field.wx[::step, ::step],
                  field.wy[::step, ::step], color="white", scale=170, width=0.0026, alpha=0.8)
    return im, q


def _dress_map(ax, extent, start, goal, goal_radius):
    xmin, xmax, ymin, ymax = extent
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), facecolor=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#2a4a63")
    ax.plot(*start, "o", color="white", mec=INK, mew=1.5, ms=11, zorder=8)
    ax.plot(*goal, "*", color="white", mec=INK, mew=1.2, ms=26, zorder=8)
    ax.add_patch(plt.Circle(goal, goal_radius, fill=False, ec="white", lw=1.3, ls=":", zorder=7))
    ax.annotate("START", start, textcoords="offset points", xytext=(12, -14), color="white",
                fontsize=9, weight="bold", path_effects=TEXT_GLOW, zorder=9)
    ax.annotate("DESTINATION", goal, textcoords="offset points", xytext=(-18, 16), color="white",
                fontsize=9, weight="bold", ha="center", path_effects=TEXT_GLOW, zorder=9)


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prefs = list(args.prefs)
    start, goal = np.array(args.start), np.array(args.goal)

    paths = {n: os.path.join(MODEL_DIR, f"{args.tag_prefix}_{n}.zip") for n in prefs}
    for spec in (args.models or []):
        n, _, path = spec.partition("=")
        paths[n] = path

    if args.scenario == "drift":
        build = lambda f: drifting_weather(f, seed=args.field_seed, drift=tuple(args.drift))
    else:
        build = BUILDERS[args.scenario]
    slices = []
    for i in range(args.slices):
        frac = i / (args.slices - 1)
        h = frac * args.window_h
        slices.append((f"2026-01-01T{int(h):02d}:{int((h % 1) * 60):02d}", build(frac)))
    field0 = slices[0][1]
    goal_radius = ShipEnv(field0).goal_radius
    span = np.linalg.norm(goal - start)
    width = field0.extent[1] - field0.extent[0]
    print(f"scenario {args.scenario}: start {start} -> destination {goal}, "
          f"{span:.1f} units apart = {100 * span / width:.0f}% of the map width")

    par = {n: P.params(n) for n in prefs}
    policies = {}
    for n in prefs:
        model, obs_cfg = load_model(paths[n])
        host = ShipEnv(field0, params=par[n])
        host.goal = goal.astype(np.float64)
        policies[n] = agent_policy(model, obs_cfg, host, wrap_wind_obs(host, obs_cfg))
        print(f"  {n:<9s} <- {paths[n]}")

    # Probe once to learn how long the SLOWEST ship takes, then stretch the weather window so
    # the whole race happens inside it (otherwise the thrifty ship sails past the last slice).
    probe = EvolvingWind(slices, 3600.0)
    voyage = max(simulate(policies[n], start, goal, probe, par[n], goal_radius,
                          args.max_steps)["t_model"] for n in prefs)
    sec_per_tu = args.window_h * 3600.0 / max(voyage, 1e-3)
    evolving = EvolvingWind(slices, sec_per_tu)
    print(f"slowest voyage {voyage:.2f} model time units -> the {args.window_h:g} h window "
          f"covers the whole race")

    runs = {}
    for n in prefs:
        r = simulate(policies[n], start, goal, evolving, par[n], goal_radius, args.max_steps,
                     refresh_min=args.update_every_min)
        r["fuel"] = fuel_curve(r["actions"], par[n])
        runs[n] = r
        print(f"  {PLAIN[n]:<11s} {'ARRIVED' if r['success'] else ('LEFT THE MAP' if r['oob'] else 'still out there')}"
              f"  {r['hours']:5.1f} h   fuel {r['fuel'][-1]:7.1f}")

    fuel_max = max(r["fuel"][-1] for r in runs.values()) or 1.0
    winner_t = min((r["hours"] for r in runs.values() if r["success"]), default=None)
    leanest = min((r["fuel"][-1] for r in runs.values() if r["success"]), default=None)
    if winner_t and leanest:
        print(f"\nheadline: earliest arrival {winner_t:.1f} h, leanest crossing {leanest:.0f} fuel "
              f"({fuel_max / leanest:.1f}x less than the thirstiest)")

    animate(evolving, start, goal, goal_radius, runs, par, prefs, args, fuel_max)
    panels(evolving, start, goal, goal_radius, runs, par, prefs, args)


def animate(evolving, start, goal, goal_radius, runs, par, prefs, args, fuel_max):
    n_steps = max(len(r["traj"]) for r in runs.values())
    frames = (n_steps + args.stride - 1) // args.stride + int(args.hold_s * args.fps)

    fig = plt.figure(figsize=(13.2, 7.4), facecolor=INK, dpi=args.dpi)
    # the wind scale is a horizontal strip UNDER the map: as a vertical bar beside it, its tick
    # labels run straight into the side panel at any column width that leaves the map usable
    gs = fig.add_gridspec(2, 2, width_ratios=[1.60, 1.02], height_ratios=[1.0, 0.045],
                          wspace=0.06, hspace=0.07, left=0.015, right=0.985, top=0.86, bottom=0.06)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[1, 0])
    side = fig.add_subplot(gs[:, 1])
    side.set_facecolor(INK)
    side.set_axis_off()

    im, q = _paint(ax, evolving.at(0.0), args.ms_per_unit)
    _dress_map(ax, evolving.fields[0].extent, start, goal, goal_radius)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    ticks, labels = wind_scale_ticks(args.ms_per_unit)
    cb.set_ticks(ticks)
    cb.set_ticklabels(labels)
    cb.set_label("wind speed (m/s)   -   calm on the left, storm on the right",
                 color="#9fc5dd", fontsize=9, labelpad=2)
    cb.ax.xaxis.set_tick_params(color="white", labelsize=8)
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="white")
    cb.outline.set_edgecolor("#2a4a63")

    lines, dots = {}, {}
    for n in prefs:
        (lines[n],) = ax.plot([], [], color=NEON[n], lw=3.0, zorder=6, path_effects=GLOW,
                              solid_capstyle="round")
        # arrival is marked by giving the dot a white ring, not a label: all three ships end up
        # inside the same small disc, so three "ARRIVED" labels would pile onto each other and
        # onto the DESTINATION caption. The side panel carries the arrival time in words.
        (dots[n],) = ax.plot([], [], "o", color=NEON[n], mec=INK, mew=1.6, ms=14, zorder=9)

    fig.text(0.015, 0.955, "Three ships. Same sea. Same destination.", color="white",
             fontsize=19, weight="bold", va="top")
    fig.text(0.015, 0.905, "They were given different instructions - and the wind keeps changing.",
             color="#9fc5dd", fontsize=12.5, va="top")
    clock = fig.text(0.985, 0.955, "", color="white", fontsize=17, weight="bold",
                     family="monospace", ha="right", va="top")

    # side panel: one card per ship, with a fuel bar that fills as it sails
    rows = {}
    for i, n in enumerate(prefs):
        y = 0.93 - i * 0.30
        side.text(0.0, y, PLAIN[n], color=NEON[n], fontsize=15, weight="bold",
                  transform=side.transAxes, va="top")
        side.text(0.0, y - 0.055, PLAIN_SUB[n], color="#9fc5dd", fontsize=10.5,
                  transform=side.transAxes, va="top", style="italic")
        side.add_patch(plt.Rectangle((0.0, y - 0.155), 0.86, 0.045, transform=side.transAxes,
                                     fc="#12293b", ec="#2a4a63", lw=0.8, zorder=2))
        bar = plt.Rectangle((0.0, y - 0.155), 0.0, 0.045, transform=side.transAxes,
                            fc=NEON[n], ec="none", zorder=3)
        side.add_patch(bar)
        label = side.text(0.0, y - 0.175, "", color="white", fontsize=11,
                          family="monospace", transform=side.transAxes, va="top")
        rows[n] = (bar, label)
    side.text(0.0, 0.015, "bars: fuel burned, as a share of the thirstiest ship",
              color="#6f93ab", fontsize=9.5, transform=side.transAxes, va="bottom")

    def update(k):
        i = min(k * args.stride, n_steps - 1)
        t_model = i * par[prefs[0]].dt
        if k % max(args.map_every, 1) == 0:
            field = evolving.at((k - k % max(args.map_every, 1)) * args.stride * par[prefs[0]].dt)
            im.set_array(field.speed.T.ravel())
            q.set_UVC(field.wx[::5, ::5], field.wy[::5, ::5])
        for n in prefs:
            r = runs[n]
            j = min(i, len(r["traj"]) - 1)
            arrived = r["success"] and j >= len(r["traj"]) - 1
            lines[n].set_data(r["traj"][:j + 1, 0], r["traj"][:j + 1, 1])
            dots[n].set_data([r["traj"][j, 0]], [r["traj"][j, 1]])
            dots[n].set_markeredgecolor("white" if arrived else INK)
            dots[n].set_markersize(16 if arrived else 14)
            fuel = r["fuel"][min(j, len(r["fuel"]) - 1)]
            bar, label = rows[n]
            bar.set_width(0.86 * fuel / fuel_max)
            hours = r["hours"] if arrived else evolving.real_hours(t_model)
            label.set_text(f"{'ARRIVED  ' if arrived else 'sailing  '}"
                           f"{hours:5.1f} h   {100 * fuel / fuel_max:3.0f}% fuel")
        clock.set_text(f"+{evolving.real_hours(t_model):4.1f} h")
        return [im, q]

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    out = os.path.join(OUTPUT_DIR, f"race_{args.scenario}{args.tag}.gif")
    ani.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    mb = os.path.getsize(out) / 1e6
    if args.gif_colors:
        mb2 = shrink_gif(out, args.gif_colors) / 1e6
        print(f"saved {out}  ({mb:.1f} MB -> {mb2:.1f} MB at {args.gif_colors} colours, "
              f"{frames} frames)")
    else:
        print(f"saved {out}  ({mb:.1f} MB, {frames} frames)")


def panels(evolving, start, goal, goal_radius, runs, par, prefs, args):
    """Four stills across the voyage, for slides where a GIF will not play."""
    n_steps = max(len(r["traj"]) for r in runs.values())
    picks = [0, n_steps // 3, 2 * n_steps // 3, n_steps - 1]
    dt = par[prefs[0]].dt
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.9), facecolor=INK)
    for ax, i in zip(axes, picks):
        _paint(ax, evolving.at(i * dt), args.ms_per_unit, step=6)
        _dress_map(ax, evolving.fields[0].extent, start, goal, goal_radius)
        for n in prefs:
            r = runs[n]
            j = min(i, len(r["traj"]) - 1)
            ax.plot(r["traj"][:j + 1, 0], r["traj"][:j + 1, 1], color=NEON[n], lw=2.6,
                    path_effects=GLOW, solid_capstyle="round")
            ax.plot(r["traj"][j, 0], r["traj"][j, 1], "o", color=NEON[n], mec=INK, ms=10)
        ax.set_title(f"+{evolving.real_hours(i * dt):.1f} h", color="white", fontsize=14, pad=8)
    handles = [plt.Line2D([], [], color=NEON[n], lw=3.2, label=f"{PLAIN[n]} - {PLAIN_SUB[n]}")
               for n in prefs]
    leg = fig.legend(handles=handles, loc="lower center", ncol=len(prefs), frameon=False, fontsize=12)
    for t in leg.get_texts():
        t.set_color("white")
    fig.suptitle("Three ships, same sea, same destination - different instructions",
                 color="white", fontsize=19, weight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    out = os.path.join(OUTPUT_DIR, f"race_{args.scenario}{args.tag}_panels.png")
    fig.savefig(out, dpi=115, facecolor=INK)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
