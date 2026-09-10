"""
Presentation-quality figures and animations of the optimal route through designed wind fields.

Solves the DP optimum on each scenario from `scenarios.py`, measures how far the optimum
departs from the straight line (so we can say quantitatively that the route is non-trivial),
and renders it in a weather-graphics style: speed shading plus streamlines, glowing track.

    python showcase.py                      # all scenarios -> one poster + per-scenario stills
    python showcase.py --animate            # also render a GIF per scenario
    python showcase.py --only jet barrier

Outputs: output/showcase/*.png and *.gif
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
from dp_baseline import ValueIterationPlanner
from scenarios import SCENARIOS, build

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output", "showcase")

TRACK = "#00f5d4"
GLOW = [pe.withStroke(linewidth=5.5, foreground="#04121f", alpha=0.9)]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None, choices=list(SCENARIOS))
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--model", default=None, help="also draw a learned policy's track for comparison")
    ap.add_argument("--nx", type=int, default=71)
    ap.add_argument("--ny", type=int, default=71)
    ap.add_argument("--nv", type=int, default=13)
    ap.add_argument("--cmap", default="magma")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def deviation(traj, start, goal):
    """Max lateral offset of the track from the straight line, as % of the direct distance."""
    d = goal - start
    L = np.linalg.norm(d)
    n = np.array([-d[1], d[0]]) / L
    off = np.abs((traj[:, :2] - start) @ n)
    return float(off.max() / L * 100.0), float(L)


def paint(ax, wind, cmap="magma", density=1.5, lw_scale=2.2):
    """Speed shading + streamlines, weather-map style. Returns the mesh."""
    sp = wind.speed
    im = ax.pcolormesh(wind.x, wind.y, sp.T, shading="gouraud", cmap=cmap,
                       vmin=0, vmax=max(10.0, float(sp.max())))
    # streamplot wants (y, x) indexed arrays
    lw = lw_scale * (sp.T / max(sp.max(), 1e-9)) ** 1.4 + 0.15
    ax.streamplot(wind.x, wind.y, wind.wx.T, wind.wy.T, color="white", density=density,
                  linewidth=lw, arrowsize=0.9, arrowstyle="-|>")
    xmin, xmax, ymin, ymax = wind.extent
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_aspect("equal")
    return im


def draw_track(ax, traj, color=TRACK, lw=3.0, ls="-", label=None):
    ax.plot(traj[:, 0], traj[:, 1], color=color, lw=lw, ls=ls, zorder=6,
            path_effects=GLOW, label=label, solid_capstyle="round")


def endpoints(ax, start, goal, goal_radius):
    ax.plot(*start, "o", color="white", mec="#04121f", mew=1.6, ms=11, zorder=8)
    ax.plot(*goal, "*", color="#ffd166", mec="#04121f", mew=1.2, ms=22, zorder=8)
    ax.add_patch(plt.Circle(goal, goal_radius, fill=False, ec="#ffd166", lw=1.3, ls=":", zorder=7))


def solve_case(name, args, params):
    wind, start, goal, title = build(name)
    env = ShipEnv(wind, params=params)
    planner = ValueIterationPlanner(wind, goal, params=params, nx=args.nx, ny=args.ny,
                                    nv=args.nv, device=args.device)
    stats = planner.solve(verbose=False)
    res = planner.rollout(env, start)
    dev, L = deviation(res["traj"], start, goal)
    print(f"{name:8s} {title:42s} ok={res['success']}  J={res['J']:5.2f}  "
          f"T={res['t']:5.2f}  detour={dev:4.1f}% of a {L:.1f}-unit leg  (solve {stats['time']:.0f}s)",
          flush=True)
    return dict(name=name, title=title, wind=wind, start=start, goal=goal,
                res=res, dev=dev, env=env, planner=planner)


def still(case, args):
    fig, ax = plt.subplots(figsize=(7.6, 7.2), facecolor="#04121f")
    ax.set_facecolor("#04121f")
    im = paint(ax, case["wind"], cmap=args.cmap)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("wind speed", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
    draw_track(ax, case["res"]["traj"], label="optimal route")
    ax.plot([case["start"][0], case["goal"][0]], [case["start"][1], case["goal"][1]],
            color="white", lw=1.3, ls=(0, (5, 4)), alpha=0.55, zorder=5, label="direct line")
    endpoints(ax, case["start"], case["goal"], case["env"].goal_radius)
    ax.set_title(f"{case['title']}\ndetour {case['dev']:.0f}% off the direct line",
                 color="white", fontsize=13, pad=12)
    for s in ax.spines.values():
        s.set_color("#2a4a63")
    ax.tick_params(colors="#9fb6c6")
    leg = ax.legend(loc="upper left", fontsize=9, framealpha=0.25, facecolor="#04121f")
    for t in leg.get_texts():
        t.set_color("white")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"{case['name']}.png")
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def poster(cases, args):
    n = len(cases)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 5.8 * nrow), facecolor="#04121f",
                             squeeze=False)
    for ax, case in zip(axes.ravel(), cases):
        ax.set_facecolor("#04121f")
        paint(ax, case["wind"], cmap=args.cmap, density=1.2, lw_scale=1.8)
        ax.plot([case["start"][0], case["goal"][0]], [case["start"][1], case["goal"][1]],
                color="white", lw=1.2, ls=(0, (5, 4)), alpha=0.5, zorder=5)
        draw_track(ax, case["res"]["traj"], lw=2.6)
        endpoints(ax, case["start"], case["goal"], case["env"].goal_radius)
        ax.set_title(f"{case['title']}  ({case['dev']:.0f}% detour)", color="white", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#2a4a63")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
        ax.set_facecolor("#04121f")
    fig.suptitle("Optimal ship routes through wind:  dashed = direct line,  cyan = optimal",
                 color="white", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(OUT_DIR, "poster.png")
    fig.savefig(out, dpi=135, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def animate(case, args, params, stride=3):
    """Lighter than the stills on purpose: GIFs go into slides, so keep them a few MB."""
    traj = case["res"]["traj"]
    n = len(traj)
    frames = (n + stride - 1) // stride
    fig, ax = plt.subplots(figsize=(5.6, 5.4), facecolor="#04121f", dpi=100)
    ax.set_facecolor("#04121f")
    paint(ax, case["wind"], cmap=args.cmap, density=1.1, lw_scale=1.6)
    ax.plot([case["start"][0], case["goal"][0]], [case["start"][1], case["goal"][1]],
            color="white", lw=1.2, ls=(0, (5, 4)), alpha=0.5, zorder=5)
    endpoints(ax, case["start"], case["goal"], case["env"].goal_radius)
    (line,) = ax.plot([], [], color=TRACK, lw=3.0, zorder=6, path_effects=GLOW,
                      solid_capstyle="round")
    (dot,) = ax.plot([], [], "o", color=TRACK, mec="#04121f", mew=1.6, ms=13, zorder=9)
    ax.set_title(case["title"], color="white", fontsize=13, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#2a4a63")
    dt = params.dt

    def update(k):
        i = min(k * stride, n - 1)
        line.set_data(traj[:i + 1, 0], traj[:i + 1, 1])
        dot.set_data([traj[i, 0]], [traj[i, 1]])
        return line, dot

    fig.tight_layout()
    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    out = os.path.join(OUT_DIR, f"{case['name']}.gif")
    ani.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    return out


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    params = ShipParams()
    names = args.only or list(SCENARIOS)
    cases = [solve_case(n, args, params) for n in names]
    cases = [c for c in cases if c["res"]["success"]]
    if not cases:
        print("no scenario solved -- check the wind strengths")
        return
    for c in cases:
        print("saved", still(c, args))
    print("saved", poster(cases, args))
    if args.animate:
        for c in cases:
            print("saved", animate(c, args, params))


if __name__ == "__main__":
    main()
