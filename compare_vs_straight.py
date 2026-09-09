"""
Compare travel time and control energy of DP, the DAgger-cloned policy, and a naive
wind-blind straight-line autopilot on held-out wind fields.

    python compare_vs_straight.py --model models/bc_v2.zip --n 30

Produces:
  output/compare_vs_straight.csv   one row per field (T, E, J, success for each method)
  output/compare_vs_straight.png   per-field bars for time and energy + aggregate summary
and prints a summary table. Energy E = sum dt*|u|^2; time T in seconds; J the total cost.
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics import ShipParams
from env import ShipEnv
from wind import WindField, generate_wind_field
from dp_baseline import ValueIterationPlanner
from baselines import straight_line_rollout, control_energy
from benchmark_dp import load_model

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
LEGACY_FIELD = os.path.join(SCRIPT_DIR, "WF.pkl")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/bc_v2.zip")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--wind", choices=["random", "legacy"], default="random")
    ap.add_argument("--nx", type=int, default=61)
    ap.add_argument("--ny", type=int, default=61)
    ap.add_argument("--nv", type=int, default=13)
    ap.add_argument("--device", default=None)
    ap.add_argument("--label", default="DAgger clone")
    return ap.parse_args()


def policy_rollout(model, env, start, goal, wind, p):
    base = env.unwrapped
    obs, info = env.reset(options=dict(start=start, goal=goal, wind=wind))
    actions = []
    for _ in range(base.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        actions.append(np.clip(np.asarray(a, dtype=np.float64), -p.u_max, p.u_max))
        if term or trunc:
            break
    actions = np.array(actions)
    return dict(J=info["J"], t=info["t"], success=info["success"], oob=info["oob"],
                E=control_energy(actions, p))


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params = ShipParams()
    model, obs_cfg = load_model(args.model)
    seeds = args.seeds if args.seeds is not None else list(range(args.n))

    rows = []
    for seed in seeds:
        wind = WindField.load_legacy(LEGACY_FIELD) if args.wind == "legacy" else generate_wind_field(seed)
        env = ShipEnv(wind, params=params)
        env.reset(seed=seed)
        start, goal = env.state[:2].copy(), env.goal.copy()

        planner = ValueIterationPlanner(wind, goal, params=params, nx=args.nx, ny=args.ny, nv=args.nv,
                                        device=args.device)
        planner.solve(verbose=False)
        dp = planner.rollout(env, start)
        dp_E = control_energy(dp["actions"], params)

        rl_env = env
        if obs_cfg is not None:
            from wind_obs import wrap_wind_obs
            rl_env = wrap_wind_obs(env, obs_cfg)
        rl = policy_rollout(model, rl_env, start, goal, wind, params)

        st = straight_line_rollout(env, start, goal, wind)

        row = dict(seed=seed,
                   dp_T=dp["t"], dp_E=dp_E, dp_J=dp["J"], dp_ok=int(dp["success"]),
                   rl_T=rl["t"], rl_E=rl["E"], rl_J=rl["J"], rl_ok=int(rl["success"]),
                   st_T=st["t"], st_E=st["E"], st_J=st["J"], st_ok=int(st["success"]))
        rows.append(row)
        print(f"seed {seed:2d}: DP T={dp['t']:5.2f} E={dp_E:6.1f} ok={dp['success']:d} | "
              f"RL T={rl['t']:5.2f} E={rl['E']:6.1f} ok={rl['success']:d} | "
              f"straight T={st['t']:5.2f} E={st['E']:7.1f} ok={st['success']:d}", flush=True)

    # ---- CSV
    out_csv = os.path.join(OUTPUT_DIR, "compare_vs_straight.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary over cases all three methods solved (fair T/E comparison)
    A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    allok = (A["dp_ok"] == 1) & (A["rl_ok"] == 1) & (A["st_ok"] == 1)
    n_all = int(allok.sum())

    def stats(prefix):
        return (A[f"{prefix}_ok"].mean() * 100,
                A[f"{prefix}_T"][allok].mean(), A[f"{prefix}_E"][allok].mean(),
                A[f"{prefix}_J"][allok].mean())

    print("\n=== summary ===")
    print(f"cases: {len(rows)}  (all three succeeded on {n_all})")
    print(f"{'method':16s} {'success':>8s} {'mean T':>8s} {'mean E':>8s} {'mean J':>8s}  (means over the {n_all} common cases)")
    for name, pre in (("DP (optimal)", "dp"), (args.label, "rl"), ("straight-line", "st")):
        sr, T, E, J = stats(pre)
        print(f"{name:16s} {sr:7.0f}% {T:8.2f} {E:8.1f} {J:8.2f}")
    # savings of DP and RL vs straight line, per case then averaged
    dT_dp = (1 - A["dp_T"][allok] / A["st_T"][allok]) * 100
    dE_dp = (1 - A["dp_E"][allok] / A["st_E"][allok]) * 100
    dT_rl = (1 - A["rl_T"][allok] / A["st_T"][allok]) * 100
    dE_rl = (1 - A["rl_E"][allok] / A["st_E"][allok]) * 100
    print(f"\nvs the straight path, median savings over the {n_all} common cases:")
    print(f"  DP           : time {np.median(dT_dp):+5.1f}%   energy {np.median(dE_dp):+5.1f}%")
    print(f"  {args.label:12s} : time {np.median(dT_rl):+5.1f}%   energy {np.median(dE_rl):+5.1f}%")

    # ---- figure
    idx = np.where(allok)[0]
    x = np.arange(len(idx))
    w = 0.27
    fig, (axT, axE, axS) = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw=dict(width_ratios=[3, 3, 1.4]))
    colors = dict(dp="#1f77b4", rl="#d62728", st="#7f7f7f")
    for ax, metric, ylabel in ((axT, "T", "travel time (s)"), (axE, "E", "control energy  Σ dt·|u|²")):
        ax.bar(x - w, A[f"dp_{metric}"][idx], w, label="DP (optimal)", color=colors["dp"])
        ax.bar(x, A[f"rl_{metric}"][idx], w, label=args.label, color=colors["rl"])
        ax.bar(x + w, A[f"st_{metric}"][idx], w, label="straight-line", color=colors["st"])
        ax.set(xlabel="held-out field", ylabel=ylabel, xticks=x)
        ax.set_xticklabels([str(A["seed"][i]) for i in idx], fontsize=7)
        ax.grid(True, axis="y", ls="--", alpha=0.4)
    axT.legend(fontsize=9)
    axT.set_title(f"Travel time  ({n_all} fields all methods solved)")
    axE.set_title("Control energy")

    # aggregate bars
    means = {m: [A[f"{p}_{m}"][allok].mean() for p in ("dp", "rl", "st")] for m in ("T", "E")}
    xa = np.arange(3)
    axS.bar(xa, means["T"], color=[colors["dp"], colors["rl"], colors["st"]])
    axS.set_xticks(xa)
    axS.set_xticklabels(["DP", args.label.split()[0], "straight"], fontsize=8)
    axS.set_title("mean travel time (s)")
    axS.grid(True, axis="y", ls="--", alpha=0.4)
    for i, v in enumerate(means["T"]):
        axS.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("DP vs learned policy vs naive straight-line autopilot on held-out wind fields", fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "compare_vs_straight.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"\nsaved {out_csv}\nsaved {out_png}")


if __name__ == "__main__":
    main()
