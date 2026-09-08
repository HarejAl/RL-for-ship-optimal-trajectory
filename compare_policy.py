"""
Plot a trained wind-aware policy against the DP baseline on the same held-out cases.

    python compare_policy.py --model models/td3_v1_best/best_model.zip --seeds 0 1 2 3

For each case (a held-out generated field with a seeded start/goal) the figure shows
the wind field, the DP trajectory and the RL trajectory with their costs. Output goes
to output/compare_<tag>.png.
"""

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from dynamics import ShipParams
from env import ShipEnv
from wind import WindField, generate_wind_field, plot_wind_field
from dp_baseline import ValueIterationPlanner
from benchmark_dp import load_model, rollout_policy

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
LEGACY_FIELD = os.path.join(SCRIPT_DIR, "WF.pkl")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--wind", choices=["random", "legacy"], default="random")
    ap.add_argument("--nx", type=int, default=61)
    ap.add_argument("--ny", type=int, default=61)
    ap.add_argument("--nv", type=int, default=13)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params = ShipParams()
    model, obs_cfg = load_model(args.model)
    legacy = WindField.load_legacy(LEGACY_FIELD) if args.wind == "legacy" else None

    n = len(args.seeds)
    ncol = min(n, 4)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.0 * nrow), squeeze=False)

    for ax, seed in zip(axes.ravel(), args.seeds):
        wind = legacy if legacy is not None else generate_wind_field(seed)
        env = ShipEnv(wind, params=params)
        env.reset(seed=seed)
        start, goal = env.state[:2].copy(), env.goal.copy()

        planner = ValueIterationPlanner(wind, goal, params=params, nx=args.nx, ny=args.ny, nv=args.nv,
                                        device=args.device)
        stats = planner.solve(verbose=False)
        dp = planner.rollout(env, start)

        rl_env = env
        if obs_cfg is not None:
            from wind_obs import wrap_wind_obs
            rl_env = wrap_wind_obs(env, obs_cfg)
        rl = rollout_policy(model, rl_env, start, goal, wind)

        plot_wind_field(ax, wind)
        ax.plot(dp["traj"][:, 0], dp["traj"][:, 1], "cyan", lw=2.2,
                label=f"DP  J={dp['J']:.2f} T={dp['t']:.2f}s ({stats['time']:.0f}s solve)")
        ok = "ok" if rl["success"] else ("oob" if rl["oob"] else "timeout")
        ax.plot(rl["traj"][:, 0], rl["traj"][:, 1], "orangered", lw=2.2, ls="--",
                label=f"RL  J={rl['J']:.2f} T={rl['t']:.2f}s ({ok}, {rl['exec_time'] * 1e3:.0f} ms)")
        ax.plot(*start, "go", ms=8)
        ax.plot(*goal, "s", color="gold", ms=9)
        gap = (rl["J"] - dp["J"]) / dp["J"] * 100 if (rl["success"] and dp["success"]) else np.nan
        ax.set(title=f"case seed {seed}   gap {gap:+.1f}%" if np.isfinite(gap) else f"case seed {seed}",
               xlabel="x", ylabel="y")
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)
        print(f"seed {seed}: DP J={dp['J']:.3f} ok={dp['success']} | RL J={rl['J']:.3f} {ok} | gap={gap:+.1f}%")

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    tag = args.tag or os.path.splitext(os.path.basename(args.model))[0]
    out = os.path.join(OUTPUT_DIR, f"compare_{tag}.png")
    fig.savefig(out, dpi=130)
    print(f"figure saved to {out}")
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg"):
        plt.show()


if __name__ == "__main__":
    main()
