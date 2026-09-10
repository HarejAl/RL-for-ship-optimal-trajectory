"""
Fetch a real 10 m wind field from Open-Meteo (no API key), plot it, and solve the DP
trajectory on it -- a demonstration that real forecast data drops into the same pipeline.

    python real_wind_demo.py                         # Ligurian Sea, first forecast hour
    python real_wind_demo.py --lat 36 40 --lon 12 18 --hour 6 --max-speed 10

Output: output/real_wind_<tag>.png  (wind field + DP trajectory + value function)
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics import ShipParams
from env import ShipEnv
from wind import WindField, plot_wind_field
from dp_baseline import ValueIterationPlanner

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, nargs=2, default=[43.0, 44.2], help="latitude range (deg)")
    ap.add_argument("--lon", type=float, nargs=2, default=[8.2, 9.8], help="longitude range (deg)")
    ap.add_argument("--nx", type=int, default=28)
    ap.add_argument("--ny", type=int, default=28)
    ap.add_argument("--hour", type=int, default=0, help="forecast hour index")
    ap.add_argument("--forecast-days", type=int, default=2)
    ap.add_argument("--model", default=None, help="Open-Meteo model id, e.g. ecmwf_ifs025")
    ap.add_argument("--domain-size", type=float, default=10.0, help="model units across the longer side")
    ap.add_argument("--ref-speed", type=float, default=15.0, help="m/s that maps to wind_ref (=10) units")
    ap.add_argument("--max-speed", type=float, default=None, help="instead rescale each field's peak to this")
    ap.add_argument("--seed", type=int, default=0, help="seed for the start/goal pair")
    ap.add_argument("--solve", action="store_true", help="also solve and draw the DP trajectory")
    ap.add_argument("--dp-nx", type=int, default=61)
    ap.add_argument("--dp-ny", type=int, default=61)
    ap.add_argument("--tag", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params = ShipParams()

    print(f"fetching Open-Meteo 10 m wind  lat={args.lat} lon={args.lon}  {args.nx}x{args.ny} grid ...")
    wind, times = WindField.from_openmeteo(
        args.lat, args.lon, nx=args.nx, ny=args.ny, hour=args.hour,
        forecast_days=args.forecast_days, model=args.model, domain_size=args.domain_size,
        ref_speed=None if args.max_speed else args.ref_speed, max_speed=args.max_speed,
        return_times=True,
    )
    m = wind.meta
    when = m["time"] or f"hour {args.hour}"
    sp = wind.speed
    print(f"forecast time {when} UTC   model={m['model']}")
    print(f"box {m['box_km'][0]}x{m['box_km'][1]} km  ->  domain span {m['domain_span']} units")
    print(f"scales: {m['km_per_unit']} km/unit, {m['ms_per_unit']} (m/s)/unit   "
          f"wind speed min/mean/max = {sp.min():.1f}/{sp.mean():.1f}/{sp.max():.1f} units "
          f"= {sp.min()*m['ms_per_unit']:.1f}/{sp.mean()*m['ms_per_unit']:.1f}/{sp.max()*m['ms_per_unit']:.1f} m/s")

    ncol = 2 if args.solve else 1
    fig, axes = plt.subplots(1, ncol, figsize=(7.5 * ncol, 6.6), squeeze=False)
    ax = axes[0, 0]
    im = plot_wind_field(ax, wind, quiver_step=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="wind speed" + ("" if args.raw else " (rescaled)"))
    ax.set(title=f"Open-Meteo 10 m wind  {when} UTC\nlat {args.lat}, lon {args.lon}", xlabel="x", ylabel="y")

    if args.solve:
        # keep start/goal inside the real-data span on both axes (the box may be non-square)
        usable = min(wind.meta["domain_span"])
        env = ShipEnv(wind, params=params, spawn_box=(0.5, usable - 0.5))
        env.reset(seed=args.seed)
        start, goal = env.state[:2].copy(), env.goal.copy()
        planner = ValueIterationPlanner(wind, goal, params=params, nx=args.dp_nx, ny=args.dp_ny)
        stats = planner.solve(verbose=False)
        res = planner.rollout(env, start)
        print(f"DP: solve {stats['time']:.1f}s  success={res['success']}  J={res['J']:.2f}  T={res['t']:.2f}s")
        ax.plot(res["traj"][:, 0], res["traj"][:, 1], "cyan", lw=2.4, label="DP trajectory")
        ax.plot(*start, "o", color="white", mec="black", ms=9, label="start")
        ax.plot(*goal, "s", color="gold", mec="black", ms=10, label="goal")
        ax.legend(loc="upper left", fontsize=8)

        ax2 = axes[0, 1]
        V = planner.value_slice()
        V = np.where(V >= 0.95 * planner.oob_cost, np.nan, V)
        cf = ax2.contourf(planner.xs.cpu().numpy(), planner.ys.cpu().numpy(), V.T, levels=30, cmap="magma")
        fig.colorbar(cf, ax=ax2, fraction=0.046, pad=0.04, label="cost-to-go V")
        ax2.plot(res["traj"][:, 0], res["traj"][:, 1], "cyan", lw=2.2)
        ax2.plot(*start, "o", color="white", mec="black", ms=9)
        ax2.plot(*goal, "s", color="gold", mec="black", ms=10)
        ax2.set_aspect("equal")
        ax2.set(title=f"DP value function (J={res['J']:.2f}, T={res['t']:.2f}s)", xlabel="x", ylabel="y")

    fig.tight_layout()
    tag = args.tag or f"lat{args.lat[0]:.0f}_{args.lat[1]:.0f}_h{args.hour}"
    out = os.path.join(OUTPUT_DIR, f"real_wind_{tag}.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
