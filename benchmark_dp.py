"""
Benchmark harness: solve many (wind field, start, goal) cases with the DP baseline
and, optionally, roll out a stable-baselines3 policy on exactly the same cases.

This produces the numbers a paper needs: per-case cost J, travel time, success,
DP solve time, and the optimality gap of the learned policy.

Examples
--------
    python benchmark_dp.py --n-cases 20 --wind random
    python benchmark_dp.py --n-cases 20 --wind legacy --model trained_model.zip

Output: output/benchmark_<tag>.csv (one row per case) and a printed summary.
"""

import argparse
import csv
import os
import time

import numpy as np

from dynamics import ShipParams
from env import ShipEnv
from wind import WindField, generate_wind_field
from dp_baseline import ValueIterationPlanner

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
LEGACY_FIELD = os.path.join(SCRIPT_DIR, "WF.pkl")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-cases", type=int, default=10)
    ap.add_argument("--wind", choices=["legacy", "random"], default="random",
                    help="legacy: same field, different start/goal; random: a new field per case")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nx", type=int, default=61)
    ap.add_argument("--ny", type=int, default=61)
    ap.add_argument("--nv", type=int, default=13)
    ap.add_argument("--n-act", type=int, default=5)
    ap.add_argument("--exec-n-act", type=int, default=9)
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=None, help="path to an SB3 TD3 model to evaluate on the same cases")
    ap.add_argument("--tag", default=None)
    return ap.parse_args()


def rollout_policy(model, env, start, goal, wind):
    obs, info = env.reset(options=dict(start=start, goal=goal, wind=wind))
    t0 = time.perf_counter()
    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return dict(J=info["J"], t=info["t"], success=info["success"], exec_time=time.perf_counter() - t0)


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params = ShipParams()
    legacy = WindField.load_legacy(LEGACY_FIELD) if args.wind == "legacy" else None
    model = None
    if args.model:
        from stable_baselines3 import TD3
        model = TD3.load(args.model, device="cpu")

    rows = []
    for k in range(args.n_cases):
        case_seed = args.seed * 10_000 + k
        wind = legacy if legacy is not None else generate_wind_field(case_seed)
        env = ShipEnv(wind, params=params)
        env.reset(seed=case_seed)
        start, goal = env.state[:2].copy(), env.goal.copy()

        planner = ValueIterationPlanner(wind, goal, params=params, nx=args.nx, ny=args.ny, nv=args.nv,
                                        n_act=args.n_act, exec_n_act=args.exec_n_act, device=args.device)
        stats = planner.solve(max_iter=args.max_iter, tol=args.tol, verbose=False)
        t0 = time.perf_counter()
        res = planner.rollout(env, start)
        row = dict(case=k, seed=case_seed, start_x=start[0], start_y=start[1], goal_x=goal[0], goal_y=goal[1],
                   dp_J=res["J"], dp_t=res["t"], dp_success=int(res["success"]),
                   dp_V_start=float(planner.value(start[0], start[1])),
                   dp_solve_time=stats["time"], dp_iters=stats["iterations"], dp_converged=int(stats["converged"]),
                   dp_exec_time=time.perf_counter() - t0)
        if model is not None:
            pr = rollout_policy(model, env, start, goal, wind)
            row.update(rl_J=pr["J"], rl_t=pr["t"], rl_success=int(pr["success"]), rl_exec_time=pr["exec_time"],
                       gap=(pr["J"] - res["J"]) / res["J"] if res["success"] and pr["success"] else np.nan)
        rows.append(row)
        msg = (f"case {k:3d}  start=({start[0]:.2f},{start[1]:.2f}) goal=({goal[0]:.2f},{goal[1]:.2f})  "
               f"DP: J={res['J']:.3f} t={res['t']:.2f}s ok={res['success']} solve={stats['time']:.1f}s")
        if model is not None:
            msg += f"  RL: J={pr['J']:.3f} t={pr['t']:.2f}s ok={pr['success']}"
        print(msg)

    tag = args.tag or f"{args.wind}_s{args.seed}_n{args.n_cases}"
    out_csv = os.path.join(OUTPUT_DIR, f"benchmark_{tag}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    dp_J = np.array([r["dp_J"] for r in rows])
    dp_ok = np.array([r["dp_success"] for r in rows], dtype=bool)
    solve = np.array([r["dp_solve_time"] for r in rows])
    print("\n=== summary ===")
    print(f"cases                : {len(rows)}")
    print(f"DP success rate      : {dp_ok.mean() * 100:.0f}%")
    print(f"DP cost J (succ.)    : mean {dp_J[dp_ok].mean():.3f}  median {np.median(dp_J[dp_ok]):.3f}")
    print(f"DP solve time        : mean {solve.mean():.2f} s  (total {solve.sum():.1f} s)")
    if model is not None:
        rl_ok = np.array([r["rl_success"] for r in rows], dtype=bool)
        gap = np.array([r["gap"] for r in rows])
        print(f"RL success rate      : {rl_ok.mean() * 100:.0f}%")
        both = dp_ok & rl_ok
        if both.any():
            print(f"RL optimality gap    : mean {np.nanmean(gap[both]) * 100:.1f}%  "
                  f"median {np.nanmedian(gap[both]) * 100:.1f}%  (over {both.sum()} cases solved by both)")
        print(f"RL exec time         : mean {np.mean([r['rl_exec_time'] for r in rows]):.3f} s per episode")
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
