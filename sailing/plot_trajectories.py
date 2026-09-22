"""
Grids of agent trajectories on held-out cases, DP optimum dashed for reference.

    python sailing/plot_trajectories.py                 # both figures
    python sailing/plot_trajectories.py --which ppo     # pure RL, uniform wind
    python sailing/plot_trajectories.py --which clone   # DP-taught clone, random wind fields

ppo   : models/sail/ppo_uniform.zip, 12 of the 40 uniform-wind eval cases spanning goal angles
        from dead upwind to dead downwind
clone : models/sail/clone_fields_r3.pt, 12 random-field eval cases, Windy-style background
Output: output/sailing/rl/trajectories_<which>.png
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
import torch

from viz import windy_cmap, windy_norm
from sailing.rl_env import SailRLEnv, eval_cases, RL_PARAMS
from sailing.rl_eval import sb3_policy, dp_reference
from sailing.dp_sail import SailDP
from sailing.animate_tacking import sail, BG, AGENT, GHOST, TACK

OUT_DIR = os.path.join(REPO_DIR, "output", "sailing", "rl")
KNOT_MS = 0.514444


def goal_angle_off_wind(case):
    wind, s, g = case
    wx, wy = wind(5.0, 5.0)
    b = np.arctan2(*(np.asarray(g) - np.asarray(s))[::-1])
    return float(np.rad2deg(abs(np.angle(np.exp(1j * (np.arctan2(-wy, -wx) - b))))))


def panel(ax, case, agent, ghost, label):
    wind, s, g = case
    xmin, xmax, ymin, ymax = wind.extent
    ax.set_facecolor(BG)
    ax.pcolormesh(wind.x, wind.y, wind.speed.T, shading="gouraud", cmap=windy_cmap(),
                  norm=windy_norm(RL_PARAMS.kts_per_wind_unit * KNOT_MS), zorder=0)
    X, Y = np.meshgrid(np.linspace(xmin + 0.6, xmax - 0.6, 9), np.linspace(ymin + 0.6, ymax - 0.6, 9), indexing="ij")
    wx, wy = wind(X, Y)
    ax.quiver(X, Y, wx, wy, color="white", alpha=0.55, scale=110, width=0.006, zorder=1)
    ax.plot(ghost["traj"][:, 0], ghost["traj"][:, 1], "--", color=GHOST, lw=1.4, alpha=0.8, zorder=3)
    tr = agent["traj"]
    ax.plot(tr[:, 0], tr[:, 1], color=AGENT, lw=2.2, zorder=4)
    k = np.nonzero(np.diff(agent["tacks"]) > 0)[0] + 1
    ax.plot(tr[k, 0], tr[k, 1], "o", color=TACK, ms=3.5, mec=BG, mew=0.5, zorder=5)
    ax.plot(*s, "o", color="white", mec=BG, ms=6, zorder=6)
    ax.plot(*g, "*", color="#ffd166", mec=BG, ms=12, zorder=6)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal", xticks=[], yticks=[])
    if agent["success"]:
        gap = (agent["t"] - ghost["t"]) / ghost["t"] * 100
        res = f"agent {agent['t']:.1f} h ({gap:+.0f}%), {agent['tacks'][-1]} tk"
    else:
        res = "agent did NOT arrive"
    ax.set_title(f"{label}\n{res} | DP {ghost['t']:.1f} h, {ghost['tacks'][-1]} tk", color="white", fontsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a4a63")


def figure(which, cases, labels, policy, env, suptitle):
    fig, axes = plt.subplots(3, 4, figsize=(13, 10.6), facecolor=BG)
    for ax, case, label in zip(axes.ravel(), cases, labels):
        wind, s, g = case
        agent = sail(env, policy, wind, s, g)
        dp = SailDP(wind, env.sail.polar, RL_PARAMS, np.asarray(g), nx=81, ny=81)
        dp.solve()
        dpp = dp.policy()
        ghost = sail(env, lambda e: dpp(e.sail), wind, s, g)
        panel(ax, case, agent, ghost, label)
    fig.suptitle(suptitle, color="white", fontsize=14)
    fig.text(0.5, 0.01, "cyan: agent   dashed white: DP optimum   red dots: tacks/gybes   "
             "white dot: start   star: goal   arrows: wind", color="white", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"trajectories_{which}.png")
    fig.savefig(out, dpi=120, facecolor=BG)
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=("ppo", "clone", "both"), default="both")
    ap.add_argument("--ppo-model", default=os.path.join(REPO_DIR, "models", "sail", "ppo_uniform.zip"))
    ap.add_argument("--clone-model", default=os.path.join(REPO_DIR, "models", "sail", "clone_fields_r3.pt"))
    args = ap.parse_args()

    if args.which in ("ppo", "both"):
        from stable_baselines3 import PPO
        cases = eval_cases("uniform", 40)
        order = np.argsort([goal_angle_off_wind(c) for c in cases])
        pick = order[np.linspace(0, len(order) - 1, 12).astype(int)]
        sel = [cases[i] for i in pick]
        labels = [f"goal {goal_angle_off_wind(c):.0f} deg off the wind" for c in sel]
        figure("ppo", sel, labels, sb3_policy(PPO.load(args.ppo_model, device="cpu")),
               SailRLEnv(None, n_actions=36), "Pure RL (PPO, no teacher) - uniform wind, held-out cases")

    if args.which in ("clone", "both"):
        from sailing.clone_dp import Student, student_policy, MAP_RES
        net = Student()
        net.load_state_dict(torch.load(args.clone_model, map_location="cpu"))
        net.eval()
        sel = eval_cases("fields", 12)
        figure("clone", sel, [f"random field #{i}" for i in range(12)], student_policy(net, "cpu"),
               SailRLEnv(None, map_res=MAP_RES), "DP-taught clone (DAgger) - random wind fields, held-out cases")


if __name__ == "__main__":
    main()
