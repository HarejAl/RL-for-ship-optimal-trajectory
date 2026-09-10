"""
How much does it cost to sail on a STALE forecast, and does the learned policy transfer
to real weather?  A quantitative study on real Open-Meteo forecast sequences.

For every case (ocean region x departure time x route) three ships sail:

  1. "oracle"   - DP solved on the departure forecast, sailing through weather that HOLDS.
                  This is the cost the planner promised, and the optimality reference.
  2. "stale"    - the SAME departure plan, sailing through the weather that actually evolves.
                  stale/oracle - 1  =  the price of planning once and not looking again.
  3. "policy"   - the wind-aware CNN policy (trained ONLY on synthetic random fields),
                  re-reading the live forecast each step. Never solves anything.
                  policy/oracle - 1  =  zero-shot transfer gap to real weather.

One forecast query per region is reused for every departure time and route, so the study
is cheap on the API and the expensive part is the DP solves.

    python staleness_study.py --model models/bc_t2.zip

Outputs: output/staleness_study.csv and output/staleness_study.png
"""

import argparse
import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics import ShipParams
from env import ShipEnv
from wind import WindField
from dp_baseline import ValueIterationPlanner
from benchmark_dp import load_model
from wind_obs import wrap_wind_obs
from receding_horizon_demo import EvolvingWind, simulate

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

REGIONS = {
    "N Atlantic":     ([48.0, 56.0], [-25.0, -12.0]),
    "Mid Atlantic":   ([38.0, 46.0], [-30.0, -18.0]),
    "Bay of Biscay":  ([43.5, 48.5], [-11.0, -3.0]),
    "W Mediterranean": ([37.0, 43.0], [1.0, 9.0]),
}
# routes as fractions of the usable span (start -> goal)
ROUTES = [((0.10, 0.15), (0.85, 0.85)),
          ((0.85, 0.15), (0.10, 0.85)),
          ((0.10, 0.50), (0.90, 0.50))]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/bc_t2.zip")
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--hours", type=int, default=64, help="forecast hours per region")
    ap.add_argument("--forecast-days", type=int, default=4)
    ap.add_argument("--ref-speed", type=float, default=25.0)
    ap.add_argument("--departures", type=int, nargs="+", default=[0, 12, 24])
    ap.add_argument("--dp-nx", type=int, default=61)
    ap.add_argument("--dp-ny", type=int, default=61)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--device", default=None)
    ap.add_argument("--regions", nargs="+", default=list(REGIONS))
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params = ShipParams()
    model, obs_cfg = load_model(args.model)
    rows = []
    t_start = time.perf_counter()

    for rname in args.regions:
        lat, lon = REGIONS[rname]
        print(f"\n=== {rname}  lat={lat} lon={lon} ===", flush=True)
        slices = WindField.from_openmeteo_sequence(
            lat, lon, nx=args.nx, ny=args.ny, hours=list(range(args.hours)),
            forecast_days=args.forecast_days, ref_speed=args.ref_speed)
        meta = slices[0][1].meta
        sec_per_tu = meta["km_per_unit"] * 1000.0 / meta["ms_per_unit"]
        span = min(meta["domain_span"])
        goal_radius = ShipEnv(slices[0][1]).goal_radius
        print(f"  {meta['km_per_unit']:.0f} km/unit, {meta['ms_per_unit']} (m/s)/unit, "
              f"1 model time unit = {sec_per_tu / 3600:.1f} h; span {span:.1f} units", flush=True)

        host = ShipEnv(slices[0][1], params=params)
        wrapper = wrap_wind_obs(host, obs_cfg)

        for dep in args.departures:
            if dep + 2 >= len(slices):
                continue
            sub = slices[dep:]
            evolving = EvolvingWind(sub, sec_per_tu)
            still = EvolvingWind([sub[0], sub[0]], sec_per_tu)
            for ri, (s_f, g_f) in enumerate(ROUTES):
                start = np.array([s_f[0] * span, s_f[1] * span])
                goal = np.array([g_f[0] * span, g_f[1] * span])
                planner = ValueIterationPlanner(sub[0][1], goal, params=params, nx=args.dp_nx,
                                                ny=args.dp_ny, device=args.device)
                st = planner.solve(verbose=False)

                oracle = simulate(lambda s, f: planner.act(s), start, goal, still, params,
                                  goal_radius, args.max_steps)
                stale = simulate(lambda s, f: planner.act(s), start, goal, evolving, params,
                                 goal_radius, args.max_steps)
                host.goal = goal.astype(np.float64)

                def policy(state, field):
                    host.wind = field
                    host.state = state.astype(np.float64)
                    a, _ = model.predict(wrapper.observation(None), deterministic=True)
                    return a

                pol = simulate(policy, start, goal, evolving, params, goal_radius, args.max_steps)

                row = dict(region=rname, departure_h=dep, route=ri,
                           km_per_unit=meta["km_per_unit"], hours_per_unit=sec_per_tu / 3600,
                           dp_solve_s=round(st["time"], 2),
                           oracle_ok=int(oracle["success"]), oracle_J=round(oracle["J"], 3),
                           oracle_h=round(oracle["hours"], 2),
                           stale_ok=int(stale["success"]), stale_J=round(stale["J"], 3),
                           stale_h=round(stale["hours"], 2),
                           pol_ok=int(pol["success"]), pol_J=round(pol["J"], 3),
                           pol_h=round(pol["hours"], 2))
                row["stale_cost"] = ((stale["J"] / oracle["J"] - 1) if (oracle["success"] and stale["success"]) else np.nan)
                row["policy_gap"] = ((pol["J"] / oracle["J"] - 1) if (oracle["success"] and pol["success"]) else np.nan)
                rows.append(row)
                print(f"  dep+{dep:2d}h route{ri}: oracle {'ok' if oracle['success'] else 'FAIL'} "
                      f"J={oracle['J']:5.2f} ({oracle['hours']:4.1f}h) | stale "
                      f"{'ok' if stale['success'] else 'FAIL'} J={stale['J']:5.2f} "
                      f"({row['stale_cost'] * 100:+5.1f}%) | policy "
                      f"{'ok' if pol['success'] else 'FAIL'} J={pol['J']:5.2f} "
                      f"({row['policy_gap'] * 100:+5.1f}%)", flush=True)

    out_csv = os.path.join(OUTPUT_DIR, "staleness_study.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summarise(rows, args)
    print(f"\nsaved {out_csv}   (total {(time.perf_counter() - t_start) / 60:.1f} min)")


def summarise(rows, args):
    A = {k: np.array([r[k] for r in rows], dtype=object) for k in rows[0]}
    stale_cost = np.array([r["stale_cost"] for r in rows], dtype=float)
    policy_gap = np.array([r["policy_gap"] for r in rows], dtype=float)
    ok = lambda k: np.array([r[k] for r in rows], dtype=int)
    n = len(rows)
    print("\n================ SUMMARY ================")
    print(f"cases: {n}   (regions x departure times x routes)")
    print(f"arrival rate  oracle {ok('oracle_ok').mean()*100:3.0f}%   "
          f"stale plan {ok('stale_ok').mean()*100:3.0f}%   policy {ok('pol_ok').mean()*100:3.0f}%")
    for name, v in (("cost of a STALE forecast", stale_cost), ("policy gap (zero-shot, real wind)", policy_gap)):
        v = v[np.isfinite(v)]
        if v.size:
            print(f"{name:34s} median {np.median(v)*100:+5.1f}%   mean {v.mean()*100:+5.1f}%   "
                  f"p90 {np.quantile(v, 0.9)*100:+5.1f}%   max {v.max()*100:+5.1f}%  (n={v.size})")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    for v, c, lab in ((stale_cost, "#d62728", "stale forecast"), (policy_gap, "#1f77b4", "learned policy")):
        v = v[np.isfinite(v)] * 100
        ax.hist(v, bins=12, alpha=0.62, color=c, label=f"{lab} (median {np.median(v):+.1f}%)")
    ax.axvline(0, color="black", lw=1)
    ax.set(xlabel="extra cost vs the DP optimum (%)", ylabel="cases",
           title="Cost above the planner's optimum")
    ax.legend(fontsize=8)

    ax = axes[1]
    regions = sorted({r["region"] for r in rows})
    xs = np.arange(len(regions))
    for off, (key, c, lab) in enumerate(((("stale_cost"), "#d62728", "stale forecast"),
                                         (("policy_gap"), "#1f77b4", "learned policy"))):
        med = [np.nanmedian([r[key] for r in rows if r["region"] == g]) * 100 for g in regions]
        ax.bar(xs + (off - 0.5) * 0.35, med, 0.35, color=c, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels([g.replace(" ", "\n") for g in regions], fontsize=8)
    ax.axhline(0, color="black", lw=1)
    ax.set(ylabel="median extra cost (%)", title="By region")
    ax.legend(fontsize=8)

    ax = axes[2]
    hrs = np.array([r["oracle_h"] for r in rows], dtype=float)
    sc = stale_cost * 100
    m = np.isfinite(sc)
    ax.scatter(hrs[m], sc[m], c="#d62728", s=26, label="stale forecast")
    pg = policy_gap * 100
    m2 = np.isfinite(pg)
    ax.scatter(hrs[m2], pg[m2], c="#1f77b4", s=26, alpha=0.75, label="learned policy")
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="voyage length (hours)", ylabel="extra cost (%)",
           title="Does a longer voyage punish a stale plan?")
    ax.legend(fontsize=8)

    fig.suptitle("Real Open-Meteo forecasts: the price of not looking at the map again, "
                 "and zero-shot transfer of the learned policy", fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "staleness_study.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
