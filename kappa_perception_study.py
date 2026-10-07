"""
Can a policy trained on ONE windage ratio steer ships of another, by scaling the wind it sees?

The policy (trained at kappa_ref) is shown W_eff = sqrt(kappa / kappa_ref) * W, which matches the
wind force on a ship at rest (`wind_obs.PerceivedWindWrapper`); the ship sails the true wind with
its true kappa. Compared on the same held-out cases (benchmark_dp seeds) against
  * "raw":       the same policy shown the true wind, and
  * DP solved on the true ship (the optimum).

    python kappa_perception_study.py --model models/bc_t2.zip --kappas 0.1 0.25 0.5 --n-cases 8

Output: output/kappa_perception_<tag>.csv and a summary table.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

from dynamics import ShipParams
from env import ShipEnv
from wind import generate_wind_field
from dp_baseline import ValueIterationPlanner
from benchmark_dp import load_model, rollout_policy
from wind_obs import PerceivedWindWrapper, wrap_wind_obs

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/bc_t2.zip")
    ap.add_argument("--kappa-ref", type=float, default=0.5, help="windage ratio the policy was trained on")
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--n-cases", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, obs_cfg = load_model(os.path.join(SCRIPT_DIR, args.model))
    rows = []
    for kappa in args.kappas:
        params = ShipParams(cd_air=kappa * ShipParams().cd_water)
        for k in range(args.n_cases):
            wind = generate_wind_field(k)
            env = ShipEnv(wind, params=params)
            env.reset(seed=k)
            start, goal = env.state[:2].copy(), env.goal.copy()
            t0 = time.time()
            planner = ValueIterationPlanner(wind, goal, params=params, nx=61, ny=61, nv=13, n_act=5,
                                            exec_n_act=9, device=args.device)
            planner.solve(max_iter=3000, tol=1e-4, verbose=False)
            dp = planner.rollout(env, start)
            t_dp = time.time() - t0
            raw = rollout_policy(model, wrap_wind_obs(ShipEnv(wind, params=params), obs_cfg), start, goal, wind)
            per = rollout_policy(model, wrap_wind_obs(PerceivedWindWrapper(ShipEnv(wind, params=params),
                                                                           args.kappa_ref), obs_cfg),
                                 start, goal, wind)
            gap = lambda r: (r["J"] - dp["J"]) / dp["J"] if (dp["success"] and r["success"]) else np.nan
            row = dict(kappa=kappa, case=k, dp_J=dp["J"], dp_ok=int(dp["success"]),
                       raw_J=raw["J"], raw_ok=int(raw["success"]), raw_gap=gap(raw),
                       per_J=per["J"], per_ok=int(per["success"]), per_gap=gap(per), dp_time=t_dp)
            rows.append(row)
            print(f"kappa {kappa:.2f} case {k}: DP J={dp['J']:.2f} ok={dp['success']} ({t_dp:.0f}s) | "
                  f"raw J={raw['J']:.2f} ok={raw['success']} | perceived J={per['J']:.2f} ok={per['success']}",
                  flush=True)
    tag = args.tag or os.path.splitext(os.path.basename(args.model))[0]
    out = os.path.join(OUTPUT_DIR, f"kappa_perception_{tag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'kappa':>6s} {'DP ok':>6s} | {'raw ok':>7s} {'raw gap med':>11s} | {'perc ok':>7s} {'perc gap med':>12s}")
    for kappa in args.kappas:
        r = [x for x in rows if x["kappa"] == kappa]
        f = lambda key: np.mean([x[key] for x in r])
        g = lambda key: np.nanmedian([x[key] for x in r]) * 100 if np.any(np.isfinite([x[key] for x in r])) else np.nan
        print(f"{kappa:6.2f} {f('dp_ok') * 100:5.0f}% | {f('raw_ok') * 100:6.0f}% {g('raw_gap'):10.1f}% | "
              f"{f('per_ok') * 100:6.0f}% {g('per_gap'):11.1f}%")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
