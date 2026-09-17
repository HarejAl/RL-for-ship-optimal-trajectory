"""
Ocean-scale, time-varying sailing routing: what does the time-dependent DP cost, and what
resolution does the answer actually need?

Uses a real 7-day Open-Meteo forecast over a ~600 nm North Atlantic box (cached after the
first fetch). For each DP grid it records solve time, layer storage, and -- where the layers
are stored -- the voyage time the DP policy achieves in the true evolving wind. It also runs
the isochrone router (which handles time natively) and a DP planned only on the departure
forecast, and estimates a learned policy's cost as decisions * per-decision time.

    python sailing/ocean_benchmark.py --converge          # grids with rollouts (voyage convergence)
    python sailing/ocean_benchmark.py --scale             # large grids, timing only
    python sailing/ocean_benchmark.py --plot              # figure from the saved CSV

Output: output/sailing/ocean_benchmark.csv, ocean_scaling.png, ocean_route.png
"""

import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
OUT_DIR = os.path.join(REPO_DIR, "output", "sailing")
CSV_PATH = os.path.join(OUT_DIR, "ocean_benchmark.csv")
sys.path.insert(0, REPO_DIR)

from viz import windy_cmap, windy_norm  # noqa: E402
from sailing.polar import Polar  # noqa: E402
from sailing.boat_env import SailEnv, rollout  # noqa: E402
from sailing.dp_sail import SailDP  # noqa: E402
from sailing.dp_time import SailDPTime  # noqa: E402
from sailing.isochrone import isochrone_route, route_follower  # noqa: E402
from sailing.wind_seq import WindSequence, sail_params_for, KNOT_MS  # noqa: E402

LAT, LON = (46.0, 56.0), (-30.0, -14.0)
START, GOAL = np.array([0.8, 2.0]), np.array([9.2, 8.0])


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--converge", action="store_true")
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--converge-grids", type=int, nargs="+", default=[81, 121, 161, 241, 361])
    ap.add_argument("--scale-grids", type=int, nargs="+", default=[481, 721])
    ap.add_argument("--goal-nm", type=float, default=10.0)
    ap.add_argument("--env-dt", type=float, default=0.1, help="env step (h) for rollouts")
    ap.add_argument("--decision-h", type=float, default=0.25, help="learned policy decision interval (h)")
    ap.add_argument("--policy-ms", type=float, default=1.0, help="per-decision cost of a learned policy (ms)")
    return ap.parse_args()


def setup(goal_nm, env_dt):
    seq = WindSequence.from_openmeteo(LAT, LON, days=7, nx=24, ny=24, name="natl_sail")
    base = sail_params_for(seq)
    p = sail_params_for(seq, goal_radius=goal_nm / base.nm_per_unit, dt=env_dt)
    return seq, p, Polar.synthetic()


def append_row(row):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def run_converge(args):
    import torch
    seq, p, polar = setup(args.goal_nm, args.env_dt)
    env = lambda: SailEnv(wind_fn=seq, polar=polar, params=p, max_steps=int(168 / p.dt))
    SailDPTime(seq, polar, p, GOAL, nx=81, ny=81, store=None, horizon_h=10).solve()   # warm-up

    for n in args.converge_grids:
        td = SailDPTime(seq, polar, p, GOAL, nx=n, ny=n, store="cpu16", verbose=True)
        info = td.solve()
        t0 = time.perf_counter()
        e = rollout(env(), td.policy(), START, GOAL)
        roll_s = time.perf_counter() - t0
        row = dict(kind="dp_time", grid=n, cell_nm=round(info["cell_nm"], 2), dt_h=round(info["dt"], 3),
                   layers=info["layers"], state_times_M=round(info["state_times"] / 1e6, 1),
                   solve_s=round(info["solve_s"], 2), store_gb=round(info["stored_gb"], 2),
                   voyage_h=round(e["t"], 2) if e["success"] else "", ok=int(e["success"]),
                   tacks=e["tacks"], gybes=e["gybes"], rollout_s=round(roll_s, 1))
        append_row(row)
        print(row, flush=True)
        if n == args.converge_grids[-1] or n == 241:
            np.save(os.path.join(OUT_DIR, f"ocean_traj_dp{n}.npy"), e["traj"])
        del td
        torch.cuda.empty_cache()

    # isochrone: time-varying natively
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iso = isochrone_route(polar, p, START, GOAL, wind_fn=seq, step_h=0.25, n_sectors=720, max_hours=168)
    e = rollout(env(), route_follower(iso, polar, p), START, GOAL) if iso["success"] else None
    row = dict(kind="isochrone", grid="", cell_nm="", dt_h=0.25, layers="", state_times_M="",
               solve_s=round(iso["compute_s"], 2), store_gb="",
               voyage_h=round(e["t"], 2) if e and e["success"] else "", ok=int(bool(e and e["success"])),
               tacks=e["tacks"] if e else "", gybes=e["gybes"] if e else "", rollout_s="")
    append_row(row)
    print(row, flush=True)
    if e is not None:
        np.save(os.path.join(OUT_DIR, "ocean_traj_iso.npy"), e["traj"])

    # DP that only ever saw the departure forecast
    st = SailDP(seq.at(0.0), polar, p, GOAL, nx=161, ny=161)
    info = st.solve()
    e = rollout(env(), st.policy(), START, GOAL)
    row = dict(kind="dp_departure_only", grid=161, cell_nm="", dt_h="", layers="", state_times_M="",
               solve_s=round(info["precompute_s"] + info["solve_s"], 2), store_gb="",
               voyage_h=round(e["t"], 2) if e["success"] else "", ok=int(e["success"]),
               tacks=e["tacks"], gybes=e["gybes"], rollout_s="")
    append_row(row)
    print(row, flush=True)
    np.save(os.path.join(OUT_DIR, "ocean_traj_static.npy"), e["traj"])


def run_scale(args):
    import torch
    seq, p, polar = setup(args.goal_nm, args.env_dt)
    SailDPTime(seq, polar, p, GOAL, nx=81, ny=81, store=None, horizon_h=10).solve()
    for n in args.scale_grids:
        td = None
        try:
            td = SailDPTime(seq, polar, p, GOAL, nx=n, ny=n, store=None, verbose=True)
            torch.cuda.reset_peak_memory_stats()
            info = td.solve(verbose=True, log_every=200)
            peak = torch.cuda.max_memory_allocated() / 1e9
            store_gb = td.K * td.N * (td.N_layers + 1) * 2 / 1e9
            row = dict(kind="dp_time_timing", grid=n, cell_nm=round(info["cell_nm"], 2), dt_h=round(info["dt"], 3),
                       layers=info["layers"], state_times_M=round(info["state_times"] / 1e6, 1),
                       solve_s=round(info["solve_s"], 2), store_gb=round(store_gb, 2),
                       voyage_h="", ok="", tacks="", gybes="", rollout_s=f"gpu_peak_gb={peak:.2f}")
        except RuntimeError as err:
            row = dict(kind="dp_time_timing", grid=n, cell_nm="", dt_h="", layers="", state_times_M="",
                       solve_s="", store_gb="", voyage_h="", ok="", tacks="", gybes="",
                       rollout_s=f"FAILED {type(err).__name__}: {str(err)[:80]}")
        append_row(row)
        print(row, flush=True)
        del td
        torch.cuda.empty_cache()


def load_rows():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def make_plots(args):
    seq, p, polar = setup(args.goal_nm, args.env_dt)
    rows = load_rows()
    num = lambda r, k: float(r[k]) if r.get(k) not in ("", None) else np.nan
    dp = sorted([r for r in rows if r["kind"] in ("dp_time", "dp_time_timing") and r["solve_s"]],
                key=lambda r: int(r["grid"]))
    iso = [r for r in rows if r["kind"] == "isochrone"]
    stat = [r for r in rows if r["kind"] == "dp_departure_only"]

    # ------------------------------------------------ scaling + convergence
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5.2))
    cells = np.array([num(r, "cell_nm") for r in dp])
    solve = np.array([num(r, "solve_s") for r in dp])
    store = np.array([num(r, "store_gb") for r in dp])
    a1.loglog(cells, solve, "o-", color="#00a896", lw=2.2, label="time-dependent DP (GPU)")
    for c, s_, g in zip(cells, solve, store):
        a1.annotate(f"{g:.1f} GB", (c, s_), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ref_voyage = np.nanmedian([num(r, "voyage_h") for r in dp if r["voyage_h"]]) if any(r["voyage_h"] for r in dp) else 90.0
    decisions = ref_voyage / args.decision_h
    a1.axhline(decisions * args.policy_ms / 1000, color="gray", ls="--",
               label=f"learned policy: {decisions:.0f} decisions x {args.policy_ms:g} ms")
    if iso and iso[0]["solve_s"]:
        a1.axhline(num(iso[0], "solve_s"), color="#ffb703", ls=":", lw=2, label="isochrone (720 sectors)")
    a1.invert_xaxis()
    a1.set(xlabel="DP cell size (nm)  -- finer to the right", ylabel="seconds to solve",
           title="Cost of one route, 600 nm, 7-day real forecast\n(labels: GB to store the value layers)")
    a1.grid(True, which="both", ls="--", alpha=0.35)
    a1.legend(fontsize=8.5)

    vc = [(num(r, "cell_nm"), num(r, "voyage_h")) for r in dp if r["voyage_h"]]
    if vc:
        c, v = zip(*vc)
        a2.plot(c, v, "o-", color="#00a896", lw=2.2, label="time-dependent DP")
    if iso and iso[0]["voyage_h"]:
        a2.axhline(num(iso[0], "voyage_h"), color="#ffb703", ls=":", lw=2, label="isochrone")
    if stat and stat[0]["voyage_h"]:
        a2.axhline(num(stat[0], "voyage_h"), color="#e76f51", ls="--", lw=2, label="DP on departure forecast only")
    a2.invert_xaxis()
    a2.set_xscale("log")
    a2.set(xlabel="DP cell size (nm)", ylabel="voyage time achieved (h)",
           title="Does resolution change the answer?")
    a2.grid(True, which="both", ls="--", alpha=0.35)
    a2.legend(fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "ocean_scaling.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)

    # ------------------------------------------------ the route over the evolving forecast
    trajs = {}
    for key, fname in (("time-dependent DP", None), ("isochrone", "ocean_traj_iso.npy"),
                       ("departure-forecast DP", "ocean_traj_static.npy")):
        if fname is None:
            cands = sorted([f for f in os.listdir(OUT_DIR) if f.startswith("ocean_traj_dp")],
                           key=lambda f: int(f[13:-4]))
            fname = cands[-1] if cands else None
        if fname and os.path.exists(os.path.join(OUT_DIR, fname)):
            trajs[key] = np.load(os.path.join(OUT_DIR, fname))
    if not trajs:
        return
    T = max(len(t) for t in trajs.values()) * p.dt
    snaps = np.linspace(0, T, 4)
    ms_per_unit = p.kts_per_wind_unit * KNOT_MS
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.6), facecolor="#04121f")
    colors = {"time-dependent DP": "#00f5d4", "isochrone": "#ffd166", "departure-forecast DP": "#ff5d5d"}
    for ax, t in zip(axes, snaps):
        f = seq.at(t)
        ax.set_facecolor("#04121f")
        ax.pcolormesh(f.x, f.y, f.speed.T, shading="gouraud", cmap=windy_cmap(), norm=windy_norm(ms_per_unit))
        X, Y = np.meshgrid(f.x, f.y, indexing="ij")
        ax.quiver(X[::2, ::2], Y[::2, ::2], f.wx[::2, ::2], f.wy[::2, ::2], color="white", alpha=0.8,
                  scale=90, width=0.004)
        for name, tr in trajs.items():
            k = min(int(t / p.dt), len(tr) - 1)
            ax.plot(tr[:k + 1, 0], tr[:k + 1, 1], color=colors[name], lw=2.3, ls="--" if "departure" in name else "-",
                    label=name)
            ax.plot(tr[k, 0], tr[k, 1], "o", color=colors[name], mec="black", ms=8)
        ax.plot(*START, "o", color="white", mec="black", ms=9)
        ax.plot(*GOAL, "*", color="#ffd166", mec="black", ms=18)
        ax.set_title(f"{seq.labels[min(int(t), len(seq.labels) - 1)][5:16].replace('T', ' ')}Z  (+{t:.0f} h)",
                     color="white")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.3, labelcolor="white")
    fig.suptitle(f"600 nm across the North Atlantic, real 7-day forecast ({LAT[0]:g}-{LAT[1]:g}N, "
                 f"{-LON[0]:g}-{-LON[1]:g}W)", color="white", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(OUT_DIR, "ocean_route.png")
    fig.savefig(out, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", out)


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.converge:
        run_converge(args)
    if args.scale:
        run_scale(args)
    if args.plot:
        make_plots(args)


if __name__ == "__main__":
    main()
