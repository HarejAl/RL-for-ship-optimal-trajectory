"""
Animate the agent crossing a wind field that CHANGES while it sails.

Real forecast windows are often bland (see receding_horizon_demo.py for the quantitative
version on real data). This builds a deliberately legible synthetic scenario -- a cyclone
that tracks across the route while the ship is under way -- so an audience can actually see
the map change and the agent respond to it.

The agent is handed a refreshed map every `--update-every-min` minutes and holds it until
the next update, exactly as an operational forecast feed would work.

    python evolving_scenario_demo.py --model models/bc_t2.zip
    python evolving_scenario_demo.py --scenario front --update-every-min 60

Outputs: output/evolving_<scenario>.gif and output/evolving_<scenario>_panels.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter

from dynamics import ShipParams
from env import ShipEnv
from wind import WindField
from dp_baseline import ValueIterationPlanner
from benchmark_dp import load_model
from wind_obs import wrap_wind_obs
from receding_horizon_demo import EvolvingWind, simulate
from scenarios import _grid, _background, _vortex, _jet
from viz import windy_cmap, windy_norm, wind_scale_ticks

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
GLOW = [pe.withStroke(linewidth=4.5, foreground="#04121f", alpha=0.9)]
TRACK = "#00f5d4"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/bc_t2.zip")
    ap.add_argument("--scenario", default="storm", choices=["storm", "front"],
                    help="'storm' is the presentable one; 'front' is currently not navigable (see its docstring)")
    ap.add_argument("--slices", type=int, default=49, help="map snapshots over the window")
    ap.add_argument("--window-h", type=float, default=24.0, help="real hours the window spans")
    ap.add_argument("--update-every-min", type=float, default=90.0)
    ap.add_argument("--ms-per-unit", type=float, default=2.5)
    ap.add_argument("--start", type=float, nargs=2, default=[0.8, 1.0])
    ap.add_argument("--goal", type=float, nargs=2, default=[9.2, 9.0])
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--dp-nx", type=int, default=61)
    ap.add_argument("--dp-ny", type=int, default=61)
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def moving_storm(frac, seed=11):
    """A cyclone tracking WNW->ESE across the direct route. `frac` in [0,1] over the window."""
    x, y, X, Y = _grid()
    wx, wy = _background(seed, amp=1.4)
    cx = -0.5 + 11.0 * frac          # sweeps across the domain
    cy = 9.0 - 5.0 * frac            # and drifts south
    vx, vy = _vortex(X, Y, cx, cy, 2.2, 11.0)
    return WindField(x, y, wx + vx, wy + vy)


def building_front(frac, seed=31):
    """
    A gale front that builds and advances from the north across the route.

    KNOWN LIMITATION: the front spans the full width with the destination beyond it, so the
    ship has to punch through rather than route around, and at any strength that makes the
    front worth avoiding neither ship arrives. Use `storm` for presentation material until
    this is reworked (put the goal on the near side, or leave a passable flank).
    """
    x, y, X, Y = _grid()
    wx, wy = _background(seed, amp=1.1)
    # kept below the ship's ~7.5-unit headwind limit for most of the window: the front must be
    # worth avoiding, not an impassable wall (at 11 units neither ship ever arrives)
    edge = 9.5 - 6.0 * frac                       # front edge marches south
    strength = 3.0 + 4.5 * min(1.0, frac * 1.6)   # and deepens
    f = 1.0 / (1.0 + np.exp(-(Y - edge) * 2.4))
    return WindField(x, y, wx - strength * f, wy + 0.25 * strength * f)


BUILDERS = {"storm": moving_storm, "front": building_front}


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params = ShipParams()
    start, goal = np.array(args.start), np.array(args.goal)
    build = BUILDERS[args.scenario]

    # snapshots across the window, labelled with a synthetic clock
    slices = []
    for i in range(args.slices):
        frac = i / (args.slices - 1)
        h = frac * args.window_h
        slices.append((f"2026-01-01T{int(h):02d}:{int((h % 1) * 60):02d}", build(frac)))
    sec_per_tu = None  # set below so the window covers the voyage

    # first pass with a provisional time scale to learn how long the voyage takes
    probe = EvolvingWind(slices, 3600.0)
    goal_radius = ShipEnv(slices[0][1]).goal_radius
    model, obs_cfg = load_model(args.model)
    host = ShipEnv(slices[0][1], params=params)
    host.goal = goal.astype(np.float64)
    wrapper = wrap_wind_obs(host, obs_cfg)

    def policy(state, field):
        host.wind = field
        host.state = state.astype(np.float64)
        a, _ = model.predict(wrapper.observation(None), deterministic=True)
        return a

    probe_run = simulate(policy, start, goal, probe, params, goal_radius, args.max_steps)
    voyage_tu = max(probe_run["t_model"], 1e-3)
    # stretch/squeeze time so the voyage spans the whole evolving window
    sec_per_tu = args.window_h * 3600.0 / voyage_tu
    evolving = EvolvingWind(slices, sec_per_tu)
    print(f"probe voyage {voyage_tu:.2f} model time units -> 1 unit = {sec_per_tu / 3600:.2f} h "
          f"so the {args.window_h:g} h window covers the crossing")

    run = simulate(policy, start, goal, evolving, params, goal_radius, args.max_steps,
                   refresh_min=args.update_every_min)
    print(f"agent: {'ARRIVED' if run['success'] else 'did not arrive'}  J={run['J']:.2f}  "
          f"voyage={run['hours']:.1f} h  steps={run['steps']}  "
          f"(map refreshed every {args.update_every_min:g} min)")

    # a departure plan for contrast: DP on the map available at t=0, then followed
    planner = ValueIterationPlanner(slices[0][1], goal, params=params, nx=args.dp_nx,
                                    ny=args.dp_ny, device=args.device)
    planner.solve(verbose=False)
    plan = simulate(lambda s, f: planner.act(s), start, goal, evolving, params,
                    goal_radius, args.max_steps)
    print(f"departure plan: {'ARRIVED' if plan['success'] else 'did not arrive'}  J={plan['J']:.2f}")

    animate(evolving, start, goal, goal_radius, run, plan, params, args)
    panels(evolving, start, goal, run, plan, params, args)


def _paint(ax, field, ms_per_unit, step=5):
    im = ax.pcolormesh(field.x, field.y, field.speed.T, shading="gouraud",
                       cmap=windy_cmap(), norm=windy_norm(ms_per_unit))
    X, Y = np.meshgrid(field.x, field.y, indexing="ij")
    q = ax.quiver(X[::step, ::step], Y[::step, ::step], field.wx[::step, ::step],
                  field.wy[::step, ::step], color="white", scale=170, width=0.0028, alpha=0.85)
    return im, q


def animate(evolving, start, goal, goal_radius, run, plan, params, args):
    n = max(len(run["traj"]), len(plan["traj"]))
    frames = (n + args.stride - 1) // args.stride
    f0 = evolving.at(0.0)
    fig, ax = plt.subplots(figsize=(7.4, 7.0), facecolor="#04121f")
    ax.set_facecolor("#04121f")
    im, q = _paint(ax, f0, args.ms_per_unit)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ticks, labels = wind_scale_ticks(args.ms_per_unit)
    cb.set_ticks(ticks)
    cb.set_ticklabels(labels)
    cb.set_label("wind speed (m/s)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

    ax.plot(*start, "o", color="white", mec="#04121f", mew=1.5, ms=10, zorder=8)
    ax.plot(*goal, "*", color="#ffd166", mec="#04121f", mew=1.2, ms=22, zorder=8)
    ax.add_patch(plt.Circle(goal, goal_radius, fill=False, ec="#ffd166", lw=1.3, ls=":", zorder=7))
    (l_plan,) = ax.plot([], [], color="#ff5d5d", lw=2.3, ls="--", zorder=6, path_effects=GLOW,
                        label="plan fixed at departure")
    (l_run,) = ax.plot([], [], color=TRACK, lw=3.0, zorder=6, path_effects=GLOW,
                       label="agent, map refreshed", solid_capstyle="round")
    (d_plan,) = ax.plot([], [], "o", color="#ff5d5d", mec="#04121f", mew=1.4, ms=11, zorder=9)
    (d_run,) = ax.plot([], [], "o", color=TRACK, mec="#04121f", mew=1.4, ms=13, zorder=9)
    xmin, xmax, ymin, ymax = f0.extent
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#2a4a63")
    leg = ax.legend(loc="upper left", fontsize=9, framealpha=0.3, facecolor="#04121f")
    for t in leg.get_texts():
        t.set_color("white")
    clock = ax.text(0.5, 1.015, "", transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=12, family="monospace", weight="bold", color="white")
    ax.set_title(f"The map changes while the ship sails - agent re-fed every "
                 f"{args.update_every_min:g} min", color="white", fontsize=12.5, pad=26)

    at = run["agent_times"]

    def update(k):
        i = k * args.stride
        t_model = i * params.dt
        t_shown = float(at[min(i, len(at) - 1)])   # the map the agent is holding
        field = evolving.at(t_shown)
        im.set_array(field.speed.T.ravel())
        q.set_UVC(field.wx[::5, ::5], field.wy[::5, ::5])
        for res, line, dot in ((plan, l_plan, d_plan), (run, l_run, d_run)):
            j = min(i, len(res["traj"]) - 1)
            line.set_data(res["traj"][:j + 1, 0], res["traj"][:j + 1, 1])
            dot.set_data([res["traj"][j, 0]], [res["traj"][j, 1]])
        clock.set_text(f"+{evolving.real_hours(t_model):4.1f} h")
        return [im, q, l_plan, l_run, d_plan, d_run, clock]

    fig.tight_layout()
    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    out = os.path.join(OUTPUT_DIR, f"evolving_{args.scenario}.gif")
    ani.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"saved {out}")


def panels(evolving, start, goal, run, plan, params, args):
    n = len(run["traj"])
    picks = [0, n // 3, 2 * n // 3, n - 1]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4), facecolor="#04121f")
    for ax, i in zip(axes, picks):
        t_model = i * params.dt
        im, _ = _paint(ax, evolving.at(t_model), args.ms_per_unit, step=6)
        ax.set_facecolor("#04121f")
        for res, c, ls, lw in ((plan, "#ff5d5d", "--", 2.0), (run, TRACK, "-", 2.6)):
            j = min(i, len(res["traj"]) - 1)
            ax.plot(res["traj"][:j + 1, 0], res["traj"][:j + 1, 1], color=c, ls=ls, lw=lw,
                    path_effects=GLOW)
            ax.plot(res["traj"][j, 0], res["traj"][j, 1], "o", color=c, mec="#04121f", ms=9)
        ax.plot(*start, "o", color="white", mec="#04121f", ms=8)
        ax.plot(*goal, "*", color="#ffd166", mec="#04121f", ms=18)
        ax.set_title(f"+{evolving.real_hours(t_model):.1f} h", color="white", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#2a4a63")
    fig.suptitle("Wind map evolving during the crossing:  red = plan fixed at departure,  "
                 "cyan = agent re-reading the refreshed map", color="white", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(OUTPUT_DIR, f"evolving_{args.scenario}_panels.png")
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
