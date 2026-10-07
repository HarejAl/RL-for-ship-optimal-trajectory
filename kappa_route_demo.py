"""
Does a windier-perceived map make the trained agent exploit the wind more? (PLAN.md step A3)

The trained agent (kappa_ref = 0.5) steers ships of different windage ratio kappa through the
same field, shown either the true wind ("raw") or W_eff = sqrt(kappa / kappa_ref) * W
("perceived", `wind_obs.PerceivedWindWrapper`). No DP: only the agent's routes are compared.

    python kappa_route_demo.py --model models/bc_t2.zip --n-cases 200

Outputs: output/kappa_routes.png (example maps + statistics over n cases)
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics import ShipParams
from env import ShipEnv
from wind import generate_wind_field
from benchmark_dp import load_model, rollout_policy
from wind_obs import PerceivedWindWrapper, wrap_wind_obs

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
STYLE = os.path.join(SCRIPT_DIR, "journal.mplstyle")
KAPPAS = [0.05, 0.1, 0.25, 0.5, 0.6]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/bc_t2.zip")
    ap.add_argument("--kappa-ref", type=float, default=0.5)
    ap.add_argument("--n-cases", type=int, default=200)
    ap.add_argument("--examples", type=int, default=3)
    return ap.parse_args()


def deviation(traj, start, goal):
    """Largest distance of the track from the straight start-goal line."""
    d = (goal - start) / np.linalg.norm(goal - start)
    rel = traj[:, :2] - start
    return float(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]).max())


def run_case(model, obs_cfg, k, kappa, mode, kappa_ref):
    wind = generate_wind_field(k)
    params = ShipParams(cd_air=kappa * ShipParams().cd_water)
    env = ShipEnv(wind, params=params)
    env.reset(seed=k)
    start, goal = env.state[:2].copy(), env.goal.copy()
    base = ShipEnv(wind, params=params)
    if mode == "perceived":
        base = PerceivedWindWrapper(base, kappa_ref)
    r = rollout_policy(model, wrap_wind_obs(base, obs_cfg), start, goal, wind)
    r.update(start=start, goal=goal, wind=wind, dev=deviation(r["traj"], start, goal))
    return r


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(STYLE):
        plt.style.use(STYLE)
    model, obs_cfg = load_model(os.path.join(SCRIPT_DIR, args.model))
    res = {(m, kp): [run_case(model, obs_cfg, k, kp, m, args.kappa_ref) for k in range(args.n_cases)]
           for m in ("raw", "perceived") for kp in KAPPAS}

    print(f"{'kappa':>6s} {'mode':>10s} {'success':>8s} {'deviation (succ.)':>18s} {'time (succ.)':>13s} {'J (succ.)':>10s}")
    stats = {}
    for m in ("raw", "perceived"):
        for kp in KAPPAS:
            rs = res[(m, kp)]
            ok = [r for r in rs if r["success"]]
            stats[(m, kp)] = dict(succ=len(ok) / len(rs), dev=np.mean([r["dev"] for r in ok]),
                                  t=np.mean([r["t"] for r in ok]), J=np.mean([r["J"] for r in ok]))
            s = stats[(m, kp)]
            print(f"{kp:6.2f} {m:>10s} {s['succ'] * 100:7.0f}% {s['dev']:18.3f} {s['t']:13.3f} {s['J']:10.3f}")
    # paired: cases every variant finished, perceived vs raw deviation
    both = [k for k in range(args.n_cases) if all(res[(m, kp)][k]["success"] for m in ("raw", "perceived")
                                                   for kp in KAPPAS)]
    print(f"\ncases finished by every variant: {len(both)}")
    for kp in KAPPAS:
        dr = np.array([res[("raw", kp)][k]["dev"] for k in both])
        dp = np.array([res[("perceived", kp)][k]["dev"] for k in both])
        print(f"kappa {kp:4.2f}: mean deviation raw {dr.mean():.3f}  perceived {dp.mean():.3f}  "
              f"(perceived/raw {dp.mean() / dr.mean():.2f})")

    # examples: cases where the perceived routes spread the most across kappa
    spread = sorted(both, key=lambda k: -np.ptp([res[("perceived", kp)][k]["dev"] for kp in KAPPAS]))
    ex = spread[:args.examples]
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(KAPPAS)))
    fig = plt.figure(figsize=(5.2 * (args.examples + 1), 9), layout="constrained")
    gs = fig.add_gridspec(2, args.examples + 1)
    for j, k in enumerate(ex):
        for row, m in enumerate(("perceived", "raw")):
            ax = fig.add_subplot(gs[row, j])
            r0 = res[(m, KAPPAS[0])][k]
            w = r0["wind"]
            X, Y = np.meshgrid(w.x, w.y, indexing="ij")
            ax.pcolormesh(X, Y, w.speed, cmap="Blues", vmin=0, vmax=10, shading="gouraud")
            s = 8
            ax.quiver(X[::s, ::s], Y[::s, ::s], w.wx[::s, ::s], w.wy[::s, ::s], color="0.4", scale=160, width=0.003)
            ax.plot(*np.c_[r0["start"], r0["goal"]], "k--", lw=0.8)
            for c, kp in zip(cols, KAPPAS):
                r = res[(m, kp)][k]
                ax.plot(r["traj"][:, 0], r["traj"][:, 1], color=c, lw=2, label=f"$\\kappa$={kp:g}")
            ax.plot(*r0["start"], "ko")
            ax.plot(*r0["goal"], "k*", ms=12)
            ax.set(xlim=(-1, 11), ylim=(-1, 11), aspect="equal", xticks=[], yticks=[])
            ax.set_title(f"case {k}: {'perceived wind' if m == 'perceived' else 'raw wind (control)'}", fontsize=10)
            if j == 0 and row == 0:
                ax.legend(fontsize=8, loc="upper left")
    for row, (key, lab) in enumerate((("dev", "mean max deviation from the straight line"),
                                      ("t", "mean voyage time (successes)"))):
        ax = fig.add_subplot(gs[row, -1])
        for m, ls in (("perceived", "-"), ("raw", "--")):
            ax.plot(KAPPAS, [stats[(m, kp)][key] for kp in KAPPAS], ls, marker="o", color="k", label=m)
        ax.axvline(args.kappa_ref, color="0.6", lw=0.8, ls=":")
        ax.set(xlabel=r"windage ratio $\kappa$ of the ship", ylabel=lab)
        ax.legend(fontsize=9)
    fig.suptitle(f"Trained agent ($\\kappa_{{ref}}$={args.kappa_ref:g}) on ships of different windage, "
                 f"{args.n_cases} held-out cases", fontsize=13)
    out = os.path.join(OUTPUT_DIR, "kappa_routes.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
