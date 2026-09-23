"""
Three agents, one sea: race the fast / balanced / eco agents across a changing wind map.

Built for an audience that has never heard of trajectory optimisation. The three agents are
identical networks trained on the same wind, differing only in what they were told to value:
arriving early, or arriving cheaply. They leave the same port at the same moment, sail the
same weather, and the animation lets you watch the disagreement play out - the impatient one
pulls ahead through the gale, the thrifty one hangs back and lets the wind do the work.

Nothing on screen is jargon: the readouts are hours at sea and fuel burned, the fuel bars are
scaled to the thirstiest agent so the punchline ("same trip, a quarter of the fuel") is readable
without a caption.

    python evolving_race.py                                    # the default demo
    python evolving_race.py --field-seed 26 --start 0.6 5.0 --goal 9.4 5.4 --tag _east
    python evolving_race.py --dpi 100 --stride 1 --map-every 1 # slower, smoother, ~40 MB

Start and destination sit at opposite corners, 10.8 units apart on a 12-unit map, so the voyage
is long enough for the three priorities to separate visibly.

CHOOSING A CASE. These clones arrive on roughly a third of crossings (see the multi-objective
section of the README), so a demo case has to be picked, not assumed: `--field-seed` and the
route were selected by sweeping 40 generated fields x 3 routes and keeping the cases where all
three agents arrive, in the expected order, by visibly different paths. Seeds 31 (the default,
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
from viz import windy_cmap, windy_norm, wind_scale_ticks, WindParticles
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
SUBTITLE = {
    "fixed": "They were given different instructions. The wind never changes.",
    "drift": "They were given different instructions - and the wind keeps changing.",
    "real":  "Real Open-Meteo forecast, {region}, {when} - they were given different instructions.",
}

PLAIN_SUB = {
    "fast": "get there first, fuel is cheap",
    "balanced": "a sensible compromise",
    "eco": "burn as little as possible",
}


def _calm_cmap():
    """White for calm water, slate for a gale. One hue, so it never competes with the tracks."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "calm", ["#ffffff", "#e6ecf3", "#c7d5e3", "#a3b7cd", "#7d94b0", "#5b738f"])


# Two looks. `simple` is the one to present with: a single-hue wind shade, thin arrows, no
# colour bar, no captions on the map, three strong tracks and three numbers. `rich` is the
# windy.com-style dark map with the animated particle flow - better for a screen you control,
# busier than it needs to be on a projector.
THEMES = {
    "simple": dict(bg="white", fg="#16202b", muted="#6b7280", frame="#c3ccd6",
                   track=dict(P.COLORS), cmap=_calm_cmap, norm_max=1.0,
                   particles=False, colorbar=False, subtitles=False, map_labels=False,
                   arrow_color="#7d8ea3", arrow_alpha=0.75, glow=None,
                   marker="#16202b", bar_bg="#e8ecf1", track_lw=3.4,
                   levels=6, shading="auto"),
    "rich": dict(bg=INK, fg="white", muted="#9fc5dd", frame="#2a4a63",
                 track=dict(NEON), cmap=windy_cmap, norm_max=1.0,
                 particles=True, colorbar=True, subtitles=True, map_labels=True,
                 arrow_color="white", arrow_alpha=0.85, glow=GLOW,
                 marker="white", bar_bg="#12293b", track_lw=3.0,
                 levels=None, shading="gouraud"),
}


def display_field(field, res=121, coarse_below=60):
    """
    An Open-Meteo field arrives on a 20x20 grid, which banded flat shading turns into huge
    blocks. Resample it onto a finer display grid with the field's own bilinear interpolation
    - the same interpolation the agents see, so this changes the picture and not the physics.

    Every drawing path must go through this, not only the first frame: the animation updates
    the mesh and the arrows in place, so a raw field handed to `set_array` after a resampled
    one was used to build the mesh is a silent shape mismatch (it surfaces later as an empty
    GIF).
    """
    if field.x.size >= coarse_below:
        return field
    gx = np.linspace(field.x[0], field.x[-1], res)
    gy = np.linspace(field.y[0], field.y[-1], res)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    wx, wy = field(GX, GY)
    return WindField(gx, gy, wx, wy, meta=field.meta)


class _ArrowField:
    """Static arrows behind the same `.step(field)` call the particle flow uses."""

    def __init__(self, ax, field, step, color, alpha):
        X, Y = np.meshgrid(field.x, field.y, indexing="ij")
        self.step_n = step
        self.q = ax.quiver(X[::step, ::step], Y[::step, ::step], field.wx[::step, ::step],
                           field.wy[::step, ::step], color=color, scale=170, width=0.0026,
                           alpha=alpha)

    def step(self, field):
        s = self.step_n
        self.q.set_UVC(field.wx[::s, ::s], field.wy[::s, ::s])


# Open-Meteo forecast sequences already fetched into output/cache by staleness_study.py /
# receding_horizon_demo.py. Missing ones are fetched on demand (no API key needed).
REAL_REGIONS = {
    "n_atlantic":    dict(file="wind_n_atlantic_20x20_h64_ref25.npz",
                          lat=(48.0, 56.0), lon=(-25.0, -12.0), hours=64),
    "mid_atlantic":  dict(file="wind_mid_atlantic_20x20_h64_ref25.npz",
                          lat=(38.0, 46.0), lon=(-35.0, -22.0), hours=64),
    "bay_of_biscay": dict(file="wind_bay_of_biscay_20x20_h64_ref25.npz",
                          lat=(43.5, 48.5), lon=(-11.0, -3.0), hours=64),
}


def real_slices(region, n_slices=None, nx=20, ny=20, ref_speed=25.0):
    """
    Hourly Open-Meteo 10 m wind for one sea area, as (timestamp, WindField) pairs.

    Read from output/cache when it is there, fetched once and cached otherwise. The fields
    arrive normalised by `WindField.from_openmeteo_sequence`, so a policy trained in model
    units works on them unchanged, and `meta` carries km/unit and (m/s)/unit to get back to
    physical units - which is what lets the real run keep the forecast's own clock instead of
    a made-up one.
    """
    spec = REAL_REGIONS[region]
    path = os.path.join(OUTPUT_DIR, "cache", spec["file"])
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as f:
            x, y, WX, WY = f["x"], f["y"], f["wx"], f["wy"]
            times = [str(t) for t in f["times"]]
            meta = dict(f["meta"].item())
        slices = [(times[i], WindField(x, y, WX[i], WY[i], meta=meta)) for i in range(len(times))]
    else:
        print(f"  fetching Open-Meteo {region} (no API key) ...", flush=True)
        slices = WindField.from_openmeteo_sequence(
            spec["lat"], spec["lon"], nx=nx, ny=ny, hours=list(range(spec["hours"])),
            ref_speed=ref_speed)
        f0 = slices[0][1]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, x=f0.x, y=f0.y,
                            wx=np.stack([f.wx for _, f in slices]),
                            wy=np.stack([f.wy for _, f in slices]),
                            times=np.array([t for t, _ in slices], dtype=object),
                            meta=np.array(f0.meta, dtype=object))
    if n_slices and n_slices < len(slices):
        keep = np.linspace(0, len(slices) - 1, n_slices).round().astype(int)
        slices = [slices[i] for i in keep]
    return slices


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
    ap.add_argument("--scenario", default="drift",
                    choices=["drift", "fixed", "real"] + sorted(BUILDERS),
                    help="'fixed': one generated field, unchanging. 'drift': a generated field "
                         "sliding across the domain. 'real': an Open-Meteo forecast sequence "
                         "(--region), on the forecast's own clock. All three are in the agents' "
                         "training distribution. The rest are the synthetic set pieces of "
                         "evolving_scenario_demo.py - far stronger than anything the generator "
                         "makes, so the clones thrash in them.")
    ap.add_argument("--region", default="n_atlantic", choices=list(REAL_REGIONS),
                    help="'real': which sea area to pull the forecast for")
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
    ap.add_argument("--wind-max-ms", type=float, default=None,
                    help="top of the wind colour scale in m/s; default is the peak of "
                         "the window being drawn, so weak real forecasts still show relief")
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--hold-s", type=float, default=1.6, help="seconds to hold the final frame")
    ap.add_argument("--dpi", type=int, default=76, help="animation resolution; the GIF is a full-frame "
                                                        "animated map, so this drives the file size")
    ap.add_argument("--gif-colors", type=int, default=None,
                    help="requantise the finished GIF onto one shared palette; 0 disables it. "
                         "Default depends on the style: worth 64 colours for the banded 'simple' "
                         "map, where the palette is small anyway, and a loss for 'rich', whose "
                         "full-spectrum map needs every slot (and where it also discards the "
                         "per-frame cropping Pillow already did).")
    ap.add_argument("--map-every", type=int, default=4,
                    help="repaint the wind map every N frames (the agents still move every frame). "
                         "A background that is identical between frames is what lets the GIF store "
                         "only the changed region, so this is the main lever on file size.")
    ap.add_argument("--save-runs", action="store_true",
                    help="also store the raw rollouts in output/race_runs/, so figures can be "
                         "remade later without re-simulating (see load_runs)")
    ap.add_argument("--no-render", action="store_true", help="simulate and store only, draw nothing")
    ap.add_argument("--style", choices=["simple", "rich"], default="simple",
                    help="'simple': one-hue wind shade, thin arrows, no colour bar, no map "
                         "captions - the one to present with. 'rich': the windy.com-style "
                         "dark map with the animated particle flow.")
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


def save_runs(path, runs, prefs, meta):
    """
    Store everything a figure could need, so images can be remade without re-simulating.

    One .npz per case holding, per agent, the trajectory, the thrusts, the cumulative fuel
    curve and the outcome, plus the case configuration (field seed, drift, start, goal,
    weather window) under `meta` as a JSON string. Rebuild the weather at any time with
    `drifting_weather(frac, seed=meta['field_seed'], drift=meta['drift'])`.
    """
    import json
    out = {"prefs": np.array(prefs), "meta": np.array(json.dumps(meta))}
    for n in prefs:
        r = runs[n]
        out[f"{n}/traj"] = r["traj"]
        out[f"{n}/actions"] = r["actions"]
        out[f"{n}/fuel"] = r["fuel"]
        out[f"{n}/summary"] = np.array([r["t_model"], r["hours"], r["J"], r["fuel"][-1],
                                        r["steps"], float(r["success"]), float(r["oob"])])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def load_runs(path):
    """Inverse of `save_runs`: returns (runs, prefs, meta)."""
    import json
    with np.load(path, allow_pickle=False) as f:
        prefs = [str(p) for p in f["prefs"]]
        meta = json.loads(str(f["meta"]))
        runs = {}
        for n in prefs:
            s = f[f"{n}/summary"]
            runs[n] = dict(traj=f[f"{n}/traj"], actions=f[f"{n}/actions"], fuel=f[f"{n}/fuel"],
                           t_model=float(s[0]), hours=float(s[1]), J=float(s[2]),
                           steps=int(s[4]), success=bool(s[5]), oob=bool(s[6]))
    return runs, prefs, meta


def shrink_gif(path, colors=128):
    """
    Requantise a finished GIF onto ONE shared palette.

    Worth it only when the map is already banded into few colours (the 'simple' style): the
    shared palette then costs nothing visually and roughly halves the file. On a full-spectrum
    map it backfires twice - it discards the per-frame cropping Pillow did, and the background
    greys crowd out the track colours, which at 64 colours turned the blue and the green agent
    into the same teal.
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


def _paint(ax, field, ms_per_unit, step=5, animated=True, theme=None, max_ms=None):
    """
    The wind map. `levels` bands the speed into that many flat steps instead of a smooth
    ramp - simpler to read at a glance, and it keeps the GIF honest: a smooth light gradient
    dithers into hundreds of near-identical greys, which both bloats the file and, if you
    then requantise it, eats the palette slots the tracks need (blue and green agents came
    out the same teal at 64 colours).
    """
    th = theme or THEMES["rich"]
    field = display_field(field)
    cmap = th["cmap"]()
    norm = windy_norm(ms_per_unit) if max_ms is None else windy_norm(ms_per_unit, max_ms)
    if th.get("levels"):
        from matplotlib.colors import BoundaryNorm
        norm = BoundaryNorm(np.linspace(norm.vmin, norm.vmax, th["levels"] + 1), cmap.N)
    im = ax.pcolormesh(field.x, field.y, field.speed.T, shading=th.get("shading", "gouraud"),
                       cmap=cmap, norm=norm)
    # particles need motion, so a still frame always gets arrows
    if animated and th["particles"]:
        return im, WindParticles(ax, field.extent)   # windy.com-style flow, as the sailing demos
    return im, _ArrowField(ax, field, step, th["arrow_color"], th["arrow_alpha"])


def _dress_map(ax, extent, start, goal, goal_radius, theme=None):
    th = theme or THEMES["rich"]
    xmin, xmax, ymin, ymax = extent
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), facecolor=th["bg"])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(th["frame"])
    mk, edge = th["marker"], th["bg"]
    ax.plot(*start, "o", color=mk, mec=edge, mew=1.5, ms=11, zorder=8)
    ax.plot(*goal, "*", color=mk, mec=edge, mew=1.2, ms=26, zorder=8)
    ax.add_patch(plt.Circle(goal, goal_radius, fill=False, ec=mk, lw=1.3, ls=":", zorder=7))
    if th["map_labels"]:
        ax.annotate("START", start, textcoords="offset points", xytext=(12, -14), color=th["fg"],
                    fontsize=9, weight="bold", path_effects=TEXT_GLOW, zorder=9)
        ax.annotate("DESTINATION", goal, textcoords="offset points", xytext=(-18, 16),
                    color=th["fg"], fontsize=9, weight="bold", ha="center",
                    path_effects=TEXT_GLOW, zorder=9)


def build_slices(scenario, n_slices, window_h, field_seed=31, drift=(6.0, 3.0), region=None):
    """
    The weather window: `n_slices` snapshots labelled with a clock.

    'fixed' repeats one generated field, so the map never changes;
    'drift'  slides a generated field across the domain;
    'real'   is an Open-Meteo forecast sequence, which brings its own timestamps;
    anything else is one of the synthetic set pieces of evolving_scenario_demo.py.
    """
    if scenario == "real":
        return real_slices(region or "n_atlantic", n_slices)
    if scenario == "fixed":
        build = lambda f: drifting_weather(0.0, seed=field_seed, drift=(0.0, 0.0))
    elif scenario == "drift":
        build = lambda f: drifting_weather(f, seed=field_seed, drift=tuple(drift))
    else:
        build = BUILDERS[scenario]
    out = []
    for i in range(n_slices):
        frac = i / (n_slices - 1)
        h = frac * window_h
        out.append((f"2026-01-01T{int(h):02d}:{int((h % 1) * 60):02d}", build(frac)))
    return out


def load_agents(paths, prefs, field0, goal, par):
    """policy callables for each preference, sharing one host env per agent."""
    policies = {}
    for n in prefs:
        model, obs_cfg = load_model(paths[n])
        host = ShipEnv(field0, params=par[n])
        host.goal = np.asarray(goal, dtype=np.float64)
        policies[n] = agent_policy(model, obs_cfg, host, wrap_wind_obs(host, obs_cfg))
    return policies


def run_case(policies, start, goal, slices, par, prefs, goal_radius, window_h,
             refresh_min, max_steps, sec_per_tu=None):
    """
    Race the agents through one weather window. Returns (runs, evolving, sec_per_tu).

    With no `sec_per_tu` the window is first probed at a provisional time scale to find how
    long the SLOWEST agent takes, then stretched so the whole race fits inside it - otherwise
    the thrifty agent sails off the end of the forecast. Note the probe changes the answer: the
    agents see different weather under a different time scale, so a case must always be judged
    on the second pass (and at the same `--slices` the renderer will use).

    Real forecasts pass an explicit `sec_per_tu` instead, taken from the field metadata, so the
    weather advances at the rate the forecast actually says it does.
    """
    if sec_per_tu is None:
        probe = EvolvingWind(slices, 3600.0)
        voyage = max(simulate(policies[n], start, goal, probe, par[n], goal_radius,
                              max_steps)["t_model"] for n in prefs)
        sec_per_tu = window_h * 3600.0 / max(voyage, 1e-3)
    evolving = EvolvingWind(slices, sec_per_tu)
    runs = {}
    for n in prefs:
        r = simulate(policies[n], start, goal, evolving, par[n], goal_radius, max_steps,
                     refresh_min=refresh_min)
        r["fuel"] = fuel_curve(r["actions"], par[n])
        runs[n] = r
    return runs, evolving, sec_per_tu


def route_spread(runs):
    """How far apart the fast and eco routes get, in domain units (mean and max of the
    one-sided nearest-point distance). The demo is worth watching when this is large."""
    a, b = runs["fast"]["traj"][:, :2], runs["eco"]["traj"][:, :2]
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).min(axis=1)
    return float(d.mean()), float(d.max())


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prefs = list(args.prefs)
    start, goal = np.array(args.start), np.array(args.goal)

    paths = {n: os.path.join(MODEL_DIR, f"{args.tag_prefix}_{n}.zip") for n in prefs}
    for spec in (args.models or []):
        n, _, path = spec.partition("=")
        paths[n] = path

    slices = build_slices(args.scenario, args.slices, args.window_h, args.field_seed,
                          args.drift, args.region)
    field0 = slices[0][1]
    goal_radius = ShipEnv(field0).goal_radius
    span = np.linalg.norm(goal - start)
    width = field0.extent[1] - field0.extent[0]
    print(f"scenario {args.scenario}: start {start} -> destination {goal}, "
          f"{span:.1f} units apart = {100 * span / width:.0f}% of the map width")

    # Shade against the strongest wind in THIS window, not the 0-25 m/s Windy scale: a real
    # Open-Meteo window peaking near 9 m/s lands entirely in the first band and renders blank.
    # Generated fields peak at the scale's top anyway, so this changes nothing for them.
    if args.wind_max_ms is None:
        peak = max(float(f.speed.max()) for _, f in slices) * args.ms_per_unit
        args.wind_max_ms = max(5.0, float(np.ceil(peak / 2.5) * 2.5))
        print(f"  wind shading scaled to {args.wind_max_ms:g} m/s (window peak {peak:.1f} m/s)")

    par = {n: P.params(n) for n in prefs}
    policies = load_agents(paths, prefs, field0, goal, par)
    for n in prefs:
        print(f"  {n:<9s} <- {paths[n]}")

    # A real forecast keeps its own clock: one model time unit is the time the ship needs to
    # cross one model length unit at the field's own scaling, so the weather advances exactly
    # as fast as the forecast says. Synthetic windows are stretched to fit the voyage instead.
    fixed_scale = None
    if args.scenario == "real":
        m = field0.meta
        fixed_scale = m["km_per_unit"] * 1000.0 / m["ms_per_unit"]
        print(f"  {args.region}: {m['km_per_unit']:.0f} km/unit, {m['ms_per_unit']:.1f} (m/s)/unit "
              f"-> 1 model time unit = {fixed_scale / 3600:.1f} real hours; "
              f"forecast covers {len(slices)} slices from {slices[0][0]}Z")
    runs, evolving, sec_per_tu = run_case(policies, start, goal, slices, par, prefs, goal_radius,
                                          args.window_h, args.update_every_min, args.max_steps,
                                          sec_per_tu=fixed_scale)
    for n in prefs:
        r = runs[n]
        print(f"  {PLAIN[n]:<11s} {'ARRIVED' if r['success'] else ('LEFT THE MAP' if r['oob'] else 'still out there')}"
              f"  {r['hours']:5.1f} h   fuel {r['fuel'][-1]:7.1f}")

    fuel_max = max(r["fuel"][-1] for r in runs.values()) or 1.0
    winner_t = min((r["hours"] for r in runs.values() if r["success"]), default=None)
    leanest = min((r["fuel"][-1] for r in runs.values() if r["success"]), default=None)
    if winner_t and leanest:
        print(f"\nheadline: earliest arrival {winner_t:.1f} h, leanest crossing {leanest:.0f} fuel "
              f"({fuel_max / leanest:.1f}x less than the thirstiest)")

    if args.save_runs:
        meta = dict(scenario=args.scenario, field_seed=args.field_seed, drift=list(args.drift),
                    start=list(map(float, start)), goal=list(map(float, goal)),
                    slices=args.slices, window_h=args.window_h, sec_per_tu=sec_per_tu,
                    update_every_min=args.update_every_min, goal_radius=goal_radius,
                    ms_per_unit=args.ms_per_unit, max_steps=args.max_steps,
                    cost_weights={n: [par[n].time_w, par[n].ctrl_w] for n in prefs})
        p = save_runs(os.path.join(OUTPUT_DIR, "race_runs",
                                   f"race_{args.scenario}{args.tag}.npz"), runs, prefs, meta)
        print(f"runs stored -> {p}")

    if not args.no_render:
        animate(evolving, start, goal, goal_radius, runs, par, prefs, args, fuel_max)
        panels(evolving, start, goal, goal_radius, runs, par, prefs, args)


def animate(evolving, start, goal, goal_radius, runs, par, prefs, args, fuel_max):
    n_steps = max(len(r["traj"]) for r in runs.values())
    frames = (n_steps + args.stride - 1) // args.stride + int(args.hold_s * args.fps)

    th = THEMES[args.style]
    track, glow = th["track"], th["glow"]
    fig = plt.figure(figsize=(13.2, 7.4), facecolor=th["bg"], dpi=args.dpi)
    if th["colorbar"]:
        # the wind scale is a horizontal strip UNDER the map: as a vertical bar beside it, its
        # tick labels run into the side panel at any column width that leaves the map usable
        gs = fig.add_gridspec(2, 2, width_ratios=[1.60, 1.02], height_ratios=[1.0, 0.045],
                              wspace=0.06, hspace=0.07, left=0.015, right=0.985,
                              top=0.86, bottom=0.06)
        ax = fig.add_subplot(gs[0, 0])
        side = fig.add_subplot(gs[:, 1])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.60, 1.02], wspace=0.06,
                              left=0.015, right=0.985, top=0.87, bottom=0.04)
        ax = fig.add_subplot(gs[0, 0])
        side = fig.add_subplot(gs[0, 1])
    side.set_facecolor(th["bg"])
    side.set_axis_off()

    im, q = _paint(ax, evolving.at(0.0), args.ms_per_unit, theme=th,
                   max_ms=getattr(args, 'wind_max_ms', None))
    _dress_map(ax, evolving.fields[0].extent, start, goal, goal_radius, theme=th)
    if th["colorbar"]:
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, 0]), orientation="horizontal")
        ticks, labels = wind_scale_ticks(args.ms_per_unit)
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels)
        cb.set_label("wind speed (m/s)   -   calm on the left, storm on the right",
                     color=th["muted"], fontsize=9, labelpad=2)
        cb.ax.xaxis.set_tick_params(color=th["fg"], labelsize=8)
        plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color=th["fg"])
        cb.outline.set_edgecolor(th["frame"])

    lines, dots = {}, {}
    for n in prefs:
        (lines[n],) = ax.plot([], [], color=track[n], lw=th["track_lw"], zorder=6,
                              path_effects=glow, solid_capstyle="round")
        # arrival is marked by ringing the dot, not by a label: all three agents end up inside
        # the same small disc, so three "ARRIVED" labels would pile onto each other. The side
        # panel carries the arrival time in words.
        (dots[n],) = ax.plot([], [], "o", color=track[n], mec=th["bg"], mew=1.6, ms=14, zorder=9)

    headline = ("One agent. A sea that will not hold still." if len(prefs) == 1
                else "Three agents. Same sea. Same destination.")
    fig.text(0.015, 0.965, headline, color=th["fg"], fontsize=19, weight="bold", va="top")
    fig.text(0.015, 0.915, SUBTITLE.get(args.scenario, SUBTITLE["drift"]).format(
        region=args.region.replace("_", " "), when=str(evolving.times[0])[:10]),
        color=th["muted"], fontsize=12.5, va="top")
    clock = fig.text(0.985, 0.965, "", color=th["fg"], fontsize=17, weight="bold",
                     family="monospace", ha="right", va="top")

    # side panel: one row per agent, with a fuel bar that fills as it sails
    rows = {}
    gap = 0.30 if th["subtitles"] else 0.27
    for i, n in enumerate(prefs):
        y = 0.93 - i * gap
        side.text(0.0, y, PLAIN[n], color=track[n], fontsize=16, weight="bold",
                  transform=side.transAxes, va="top")
        if th["subtitles"]:
            side.text(0.0, y - 0.055, PLAIN_SUB[n], color=th["muted"], fontsize=10.5,
                      transform=side.transAxes, va="top", style="italic")
        bar_y = y - (0.155 if th["subtitles"] else 0.105)
        side.add_patch(plt.Rectangle((0.0, bar_y), 0.86, 0.045, transform=side.transAxes,
                                     fc=th["bar_bg"], ec=th["frame"], lw=0.8, zorder=2))
        bar = plt.Rectangle((0.0, bar_y), 0.0, 0.045, transform=side.transAxes,
                            fc=track[n], ec="none", zorder=3)
        side.add_patch(bar)
        label = side.text(0.0, bar_y - 0.02, "", color=th["fg"], fontsize=11.5,
                          family="monospace", transform=side.transAxes, va="top")
        rows[n] = (bar, label)
    side.text(0.0, 0.015,
              "bar: fuel burned so far, against this agent's own total"
              if len(prefs) == 1 else "bars: fuel burned, as a share of the thirstiest agent",
              color=th["muted"], fontsize=9.5, transform=side.transAxes, va="bottom")

    def update(k):
        i = min(k * args.stride, n_steps - 1)
        t_model = i * par[prefs[0]].dt
        if k % max(args.map_every, 1) == 0:
            field = display_field(
                evolving.at((k - k % max(args.map_every, 1)) * args.stride * par[prefs[0]].dt))
            im.set_array(field.speed.T.ravel())
            q.step(field)
        for n in prefs:
            r = runs[n]
            j = min(i, len(r["traj"]) - 1)
            arrived = r["success"] and j >= len(r["traj"]) - 1
            lines[n].set_data(r["traj"][:j + 1, 0], r["traj"][:j + 1, 1])
            dots[n].set_data([r["traj"][j, 0]], [r["traj"][j, 1]])
            dots[n].set_markeredgecolor(th["fg"] if arrived else th["bg"])
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
    colors = args.gif_colors if args.gif_colors is not None else (64 if th.get("levels") else 0)
    if colors:
        mb2 = shrink_gif(out, colors) / 1e6
        print(f"saved {out}  ({mb:.1f} MB -> {mb2:.1f} MB at {colors} colours, "
              f"{frames} frames)")
    else:
        print(f"saved {out}  ({mb:.1f} MB, {frames} frames)")


def panels(evolving, start, goal, goal_radius, runs, par, prefs, args):
    """Four stills across the voyage, for slides where a GIF will not play."""
    th = THEMES[args.style]
    track = th["track"]
    n_steps = max(len(r["traj"]) for r in runs.values())
    picks = [0, n_steps // 3, 2 * n_steps // 3, n_steps - 1]
    dt = par[prefs[0]].dt
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.9), facecolor=th["bg"])
    for ax, i in zip(axes, picks):
        _paint(ax, evolving.at(i * dt), args.ms_per_unit, step=6, animated=False, theme=th,
               max_ms=getattr(args, 'wind_max_ms', None))
        _dress_map(ax, evolving.fields[0].extent, start, goal, goal_radius, theme=th)
        for n in prefs:
            r = runs[n]
            j = min(i, len(r["traj"]) - 1)
            ax.plot(r["traj"][:j + 1, 0], r["traj"][:j + 1, 1], color=track[n], lw=2.8,
                    path_effects=th["glow"], solid_capstyle="round")
            ax.plot(r["traj"][j, 0], r["traj"][j, 1], "o", color=track[n], mec=th["bg"], ms=10)
        ax.set_title(f"+{evolving.real_hours(i * dt):.1f} h", color=th["fg"], fontsize=14, pad=8)
    label = (lambda n: f"{PLAIN[n]} - {PLAIN_SUB[n]}") if th["subtitles"] else (lambda n: PLAIN[n])
    handles = [plt.Line2D([], [], color=track[n], lw=3.2, label=label(n)) for n in prefs]
    leg = fig.legend(handles=handles, loc="lower center", ncol=len(prefs), frameon=False, fontsize=13)
    for t in leg.get_texts():
        t.set_color(th["fg"])
    fig.suptitle("One agent reading a changing forecast" if len(prefs) == 1 else
                 "Three agents, same sea, same destination - different instructions",
                 color=th["fg"], fontsize=19, weight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    out = os.path.join(OUTPUT_DIR, f"race_{args.scenario}{args.tag}_panels.png")
    fig.savefig(out, dpi=115, facecolor=th["bg"])
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
