"""
Solve the optimal ship trajectory with the dynamic-programming baseline and plot it.

Examples
--------
    python run_dp_baseline.py                          # legacy WF.pkl field, random start/goal (seed 0)
    python run_dp_baseline.py --wind random --seed 7   # random generated field
    python run_dp_baseline.py --start 1 1 --goal 9 9 --nx 81 --ny 81 --nv 15 --n-act 7

Outputs go to output/dp_baseline_<tag>.png and output/dp_baseline_<tag>.npz.
"""

import argparse
import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from dynamics import ShipParams
from env import ShipEnv
from wind import WindField, generate_wind_field, plot_wind_field
from dp_baseline import ValueIterationPlanner

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
LEGACY_FIELD = os.path.join(SCRIPT_DIR, "WF.pkl")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wind", choices=["legacy", "random"], default="legacy")
    ap.add_argument("--seed", type=int, default=0, help="seed for wind generation and start/goal sampling")
    ap.add_argument("--start", type=float, nargs=2, default=None)
    ap.add_argument("--goal", type=float, nargs=2, default=None)
    ap.add_argument("--nx", type=int, default=61)
    ap.add_argument("--ny", type=int, default=61)
    ap.add_argument("--nv", type=int, default=13)
    ap.add_argument("--n-act", type=int, default=5)
    ap.add_argument("--exec-n-act", type=int, default=9)
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    ap.add_argument("--ctrl-w", type=float, default=ShipParams.ctrl_w)
    ap.add_argument("--time-w", type=float, default=ShipParams.time_w)
    ap.add_argument("--tag", default=None, help="suffix of the output files")
    ap.add_argument("--no-plot", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params = ShipParams(ctrl_w=args.ctrl_w, time_w=args.time_w)

    if args.wind == "legacy":
        wind = WindField.load_legacy(LEGACY_FIELD)
    else:
        wind = generate_wind_field(args.seed)

    env = ShipEnv(wind, params=params)
    env.reset(seed=args.seed)  # samples a start/goal pair
    start = np.array(args.start) if args.start else env.state[:2].copy()
    goal = np.array(args.goal) if args.goal else env.goal.copy()
    print(f"wind={args.wind}  start={start.round(2)}  goal={goal.round(2)}  "
          f"max wind speed={wind.speed.max():.2f}")

    planner = ValueIterationPlanner(
        wind, goal, params=params, nx=args.nx, ny=args.ny, nv=args.nv,
        n_act=args.n_act, exec_n_act=args.exec_n_act, device=args.device, verbose=True,
    )
    stats = planner.solve(max_iter=args.max_iter, tol=args.tol)

    t0 = time.perf_counter()
    res = planner.rollout(env, start)
    exec_time = time.perf_counter() - t0

    print("\n=== DP baseline result ===")
    print(f"success        : {res['success']}   (out of bounds: {res['oob']})")
    print(f"cost J         : {res['J']:.3f}")
    print(f"travel time    : {res['t']:.2f} s  ({res['steps']} steps)")
    print(f"env return     : {res['return_']:.2f}")
    print(f"V(start)       : {float(planner.value(start[0], start[1])):.3f}  (predicted J)")
    print(f"VI solve time  : {stats['time']:.2f} s  ({stats['iterations']} iterations, "
          f"precompute {stats['precompute_time']:.2f} s, {stats['n_states']:,} states x {stats['n_actions']} actions)")
    print(f"policy exec    : {exec_time:.3f} s for {res['steps']} steps")

    tag = args.tag or f"{args.wind}_s{args.seed}"
    np.savez(os.path.join(OUTPUT_DIR, f"dp_baseline_{tag}.npz"),
             traj=res["traj"], actions=res["actions"], start=start, goal=goal,
             J=res["J"], t=res["t"], success=res["success"],
             solve_time=stats["time"], iterations=stats["iterations"],
             value_slice=planner.value_slice(), xs=planner.xs.cpu().numpy(), ys=planner.ys.cpu().numpy(),
             history=np.array(planner.history))

    if args.no_plot:
        return

    traj = res["traj"]
    t = np.arange(res["steps"]) * params.dt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    im = plot_wind_field(ax, wind)
    fig.colorbar(im, ax=ax, label="wind speed")
    ax.plot(traj[:, 0], traj[:, 1], "cyan", lw=2, label="DP trajectory")
    ax.plot(*start, "go", ms=8, label="start")
    ax.plot(*goal, "s", color="gold", ms=9, label="goal")
    ax.set(title=f"Wind field and DP trajectory  (J={res['J']:.2f}, T={res['t']:.2f}s)", xlabel="x", ylabel="y")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    V = planner.value_slice()
    V = np.where(V >= 0.95 * planner.oob_cost, np.nan, V)
    cf = ax.contourf(planner.xs.cpu().numpy(), planner.ys.cpu().numpy(), V.T, levels=30, cmap="magma")
    fig.colorbar(cf, ax=ax, label="cost-to-go V(x, y, v=0)")
    ax.plot(traj[:, 0], traj[:, 1], "cyan", lw=2)
    ax.plot(*start, "go", ms=8)
    ax.plot(*goal, "s", color="gold", ms=9)
    ax.set_aspect("equal")
    ax.set(title="Value function at zero velocity", xlabel="x", ylabel="y")

    ax = axes[2]
    ax.plot(t, res["actions"][:, 0], label="ux")
    ax.plot(t, res["actions"][:, 1], label="uy")
    ax.plot(t, np.hypot(traj[1:, 2], traj[1:, 3]), "k--", label="|v|")
    ax.set(title="Control and speed", xlabel="time (s)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, f"dp_baseline_{tag}.png")
    fig.savefig(out_png, dpi=130)
    print(f"figure saved to {out_png}")
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg"):
        plt.show()


if __name__ == "__main__":
    main()
