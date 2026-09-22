"""
Animate the pure-RL sailing agent (PPO, no teacher) beating upwind, with the DP optimum as a ghost.

    python sailing/animate_tacking.py --scenario beat     # goal dead upwind: tacking is required
    python sailing/animate_tacking.py --scenario reach    # goal 65 deg off the wind: no tack needed

Both boats sail the same `SailRLEnv` (same polar, 6-min tack penalty). Drawn per frame: the
boat as an arrow along its heading, the no-go cone (+-40 deg into the wind) at the boat, the
tack points, and moving wind streaks. Output: output/sailing/rl/tacking_<scenario>.gif (+ .png)
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, REPO_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Wedge, FancyArrow

from wind import uniform_wind_field
from viz import windy_cmap, windy_norm
from sailing.rl_env import SailRLEnv, RL_PARAMS
from sailing.dp_sail import SailDP
from sailing.polar import wind_geometry

OUT_DIR = os.path.join(REPO_DIR, "output", "sailing", "rl")
KNOT_MS = 0.514444
BG = "#04121f"
AGENT, GHOST, TACK = "#00e5ff", "#ffffff", "#ff4d6d"
GLOW = [pe.Stroke(linewidth=6, foreground=BG, alpha=0.55), pe.Normal()]

SCENARIOS = {
    # wind blows towards -y (a northerly); speed in wind units (x2 kt)
    "beat": dict(wind=(0.0, -6.0), start=(5.0, 0.8), goal=(5.0, 9.2),
                 title="Goal dead upwind: the agent has to tack"),
    "reach": dict(wind=(0.0, -6.0), start=(1.2, 1.5),
                  goal=tuple(np.array([1.2, 1.5]) + 8.5 * np.array([np.cos(np.deg2rad(25)), np.sin(np.deg2rad(25))])),
                  title="Goal 65 deg off the wind: one board is enough"),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=SCENARIOS, default="beat")
    ap.add_argument("--model", default=os.path.join(REPO_DIR, "models", "sail", "ppo_uniform_snapshot.zip"))
    ap.add_argument("--n-actions", type=int, default=36)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1)
    return ap.parse_args()


def sail(env, policy, wind, start, goal, max_steps=600):
    env.reset(options=dict(wind=wind, start=np.array(start), goal=np.array(goal)))
    traj, tacks, kts = [env.sail.state.copy()], [0], [0.0]
    info = env.sail._info()
    for _ in range(max_steps):
        _, _, term, trunc, info = env.sail.step(np.array([policy(env)]))
        traj.append(env.sail.state.copy())
        tacks.append(info["tacks"] + info["gybes"])
        kts.append(info.get("speed_kts", 0.0) if env.sail.pending <= 1e-12 else 0.0)
        if term or trunc:
            break
    return dict(traj=np.array(traj), tacks=np.array(tacks), kts=np.array(kts), t=info["t"],
                success=info["success"])


def main():
    args = parse_args()
    from stable_baselines3 import PPO
    from sailing.rl_eval import sb3_policy

    sc = SCENARIOS[args.scenario]
    wind = uniform_wind_field(*sc["wind"], nx=5, ny=5)
    env = SailRLEnv(None, n_actions=args.n_actions)
    agent = sail(env, sb3_policy(PPO.load(args.model, device="cpu")), wind, sc["start"], sc["goal"])
    dp = SailDP(wind, env.sail.polar, RL_PARAMS, np.array(sc["goal"]), nx=81, ny=81)
    dp.solve()
    dpp = dp.policy()
    ghost = sail(env, lambda e: dpp(e.sail), wind, sc["start"], sc["goal"])
    for name, r in (("agent", agent), ("DP", ghost)):
        print(f"{name:>5}: {'arrived' if r['success'] else 'did NOT arrive'} in {r['t']:.1f} h, "
              f"{r['tacks'][-1]} tacks/gybes")

    p = RL_PARAMS
    ms_per_unit = p.kts_per_wind_unit * KNOT_MS
    wx, wy = sc["wind"]
    tws_kts = np.hypot(wx, wy) * p.kts_per_wind_unit
    xmin, xmax, ymin, ymax = wind.extent

    fig, ax = plt.subplots(figsize=(7.0, 7.4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.imshow(np.full((2, 2), np.hypot(wx, wy)), extent=(xmin, xmax, ymin, ymax), origin="lower",
              cmap=windy_cmap(), norm=windy_norm(ms_per_unit), alpha=0.9, zorder=0)

    # wind streaks advected with the flow
    rng = np.random.default_rng(1)
    pts = rng.uniform((xmin, ymin), (xmax, ymax), size=(140, 2))
    wdir = np.array([wx, wy]) / np.hypot(wx, wy)
    streaks = LineCollection([], colors="white", linewidths=1.2, alpha=0.35, zorder=1)
    ax.add_collection(streaks)

    ax.plot(*sc["start"], "o", color="white", mec=BG, mew=1.5, ms=10, zorder=8)
    ax.plot(*sc["goal"], "*", color="#ffd166", mec=BG, mew=1.2, ms=22, zorder=8)
    ax.add_patch(plt.Circle(sc["goal"], p.goal_radius, fill=False, ec="#ffd166", lw=1.3, ls=":", zorder=7))
    (l_ghost,) = ax.plot([], [], color=GHOST, lw=1.8, ls="--", alpha=0.7, zorder=5,
                         label=f"DP optimum ({ghost['t']:.1f} h, {ghost['tacks'][-1]} tacks)")
    (l_agent,) = ax.plot([], [], color=AGENT, lw=3.0, zorder=6, path_effects=GLOW, solid_capstyle="round",
                         label=f"RL agent, no teacher ({agent['t']:.1f} h, {agent['tacks'][-1]} tacks)")
    (d_tacks,) = ax.plot([], [], "o", color=TACK, mec=BG, mew=1.0, ms=7, zorder=7, label="tack")
    (d_ghost,) = ax.plot([], [], "o", color=GHOST, mec=BG, ms=8, alpha=0.8, zorder=8)
    cone = Wedge((0, 0), 1.3, 0, 0, color=TACK, alpha=0.22, zorder=4)
    ax.add_patch(cone)
    boat = [None]

    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#2a4a63")
    leg = ax.legend(loc="lower left", fontsize=9, framealpha=0.35, facecolor=BG, edgecolor="#2a4a63")
    for t in leg.get_texts():
        t.set_color("white")
    ax.set_title(sc["title"], color="white", fontsize=13, pad=30)
    hud = ax.text(0.5, 1.012, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=11,
                  family="monospace", color="white")
    ax.text(0.98, 0.98, f"wind {tws_kts:.0f} kt\n  from the north", transform=ax.transAxes, ha="right",
            va="top", color="white", fontsize=10, family="monospace")
    ax.text(0.98, 0.86, "shaded wedge = no-go zone\n(can't sail within 40 deg of the wind)",
            transform=ax.transAxes, ha="right", va="top", color="#ffb3c1", fontsize=8)

    n = max(len(agent["traj"]), len(ghost["traj"]))
    frames = list(range(0, n, args.stride)) + [n - 1] * (2 * args.fps)   # hold the last frame
    wind_from_deg = np.rad2deg(np.arctan2(-wy, -wx))

    def update(i):
        nonlocal pts
        pts = pts + wdir * 0.12
        pts[:, 0] = (pts[:, 0] - xmin) % (xmax - xmin) + xmin
        pts[:, 1] = (pts[:, 1] - ymin) % (ymax - ymin) + ymin
        streaks.set_segments([[q, q - wdir * 0.45] for q in pts])

        ja = min(i, len(agent["traj"]) - 1)
        jg = min(i, len(ghost["traj"]) - 1)
        tr = agent["traj"]
        l_agent.set_data(tr[:ja + 1, 0], tr[:ja + 1, 1])
        l_ghost.set_data(ghost["traj"][:jg + 1, 0], ghost["traj"][:jg + 1, 1])
        d_ghost.set_data([ghost["traj"][jg, 0]], [ghost["traj"][jg, 1]])
        k = np.nonzero(np.diff(agent["tacks"][:ja + 1]) > 0)[0] + 1
        d_tacks.set_data(tr[k, 0], tr[k, 1])

        x, y, h = tr[ja]
        cone.set_center((x, y))
        cone.set_theta1(wind_from_deg - 40)
        cone.set_theta2(wind_from_deg + 40)
        if boat[0] is not None:
            boat[0].remove()
        boat[0] = ax.add_patch(FancyArrow(x - 0.25 * np.cos(h), y - 0.25 * np.sin(h), 0.5 * np.cos(h),
                                          0.5 * np.sin(h), width=0.12, head_width=0.38, head_length=0.3,
                                          color=AGENT, ec=BG, lw=1.0, zorder=9, length_includes_head=True))
        _, twa, _ = wind_geometry(wx, wy, h, p.kts_per_wind_unit)
        t_h = min(i, len(agent["traj"]) - 1) * p.dt
        status = "ARRIVED" if i >= len(agent["traj"]) - 1 and agent["success"] else f"TWA {float(twa):3.0f} deg"
        hud.set_text(f"t = {t_h:4.1f} h   speed {agent['kts'][ja]:4.1f} kt   tacks {agent['tacks'][ja]:2d}   {status}")
        return []

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"tacking_{args.scenario}.gif")
    FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps).save(out, writer=PillowWriter(fps=args.fps))
    update(n - 1)
    fig.savefig(out.replace(".gif", ".png"), dpi=130, facecolor=BG)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
