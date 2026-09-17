"""
Sailing baselines: the polar, the two planners, and what they cost to compute.

    python sailing/demo_baselines.py            # figures
    python sailing/demo_baselines.py --bench    # also time DP vs grid size (GPU and CPU)

Outputs in output/sailing/:
    polar.png        polar diagram of the synthetic cruiser
    beat.png         80 nm dead-upwind beat: isochrone fronts, both routes, the tacks
    random_field.png both planners on a random wind field, DP value function behind
    timing.png       compute time per route (only with --bench)
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
OUT_DIR = os.path.join(REPO_DIR, "output", "sailing")
sys.path.insert(0, REPO_DIR)

from wind import uniform_wind_field, generate_wind_field  # noqa: E402
from viz import windy_cmap, windy_norm  # noqa: E402
from sailing.polar import Polar, TWA_GRID  # noqa: E402
from sailing.boat_env import SailParams, SailEnv, rollout  # noqa: E402
from sailing.isochrone import isochrone_route, route_follower  # noqa: E402
from sailing.dp_sail import SailDP  # noqa: E402

KNOT_MS = 0.514444


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7, help="random wind field for random_field.png")
    ap.add_argument("--bench", action="store_true", help="time DP against grid size on GPU and CPU")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def wind_background(ax, wf, p, step=8):
    ms_per_unit = p.kts_per_wind_unit * KNOT_MS
    im = ax.pcolormesh(wf.x, wf.y, wf.speed.T, shading="gouraud", cmap=windy_cmap(),
                       norm=windy_norm(ms_per_unit))
    X, Y = np.meshgrid(wf.x, wf.y, indexing="ij")
    ax.quiver(X[::step, ::step], Y[::step, ::step], wf.wx[::step, ::step], wf.wy[::step, ::step],
              color="white", alpha=0.75, scale=160, width=0.003)
    return im


def kt_colorbar(fig, im, ax, p):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    kts = np.arange(0, 45, 10)
    cb.set_ticks(kts / p.kts_per_wind_unit)
    cb.set_ticklabels([f"{k}" for k in kts])
    cb.set_label("true wind speed (kt)")


def fig_polar(polar):
    fig = plt.figure(figsize=(6.6, 6.6))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    for tws, c in ((6, "#3961a0"), (12, "#359f35"), (20, "#9f7f3a")):
        v = polar.speed(TWA_GRID, np.full_like(TWA_GRID, float(tws)))
        th = np.deg2rad(TWA_GRID)
        ax.plot(th, v, color=c, lw=2.2, label=f"TWS {tws} kt")
        ax.plot(-th, v, color=c, lw=2.2)
        up, dn = polar.best_vmg(tws, True), polar.best_vmg(tws, False)
        for twa, _ in (up, dn):
            ax.plot([np.deg2rad(twa)], [polar.speed(twa, tws)], "o", color=c, ms=6)
    ax.set_thetamin(-180)
    ax.set_thetamax(180)
    ax.set_title("Synthetic cruiser polar (boat speed, kt)\ndots = best upwind / downwind VMG angle", pad=20)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.08, -0.08))
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "polar.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_beat(polar, p, device):
    wf = uniform_wind_field(wx=0.0, wy=-6.0)
    s, g = np.array([5.0, 1.0]), np.array([5.0, 9.0])
    iso = isochrone_route(polar, p, s, g, wind=wf, step_h=0.1)
    ie = rollout(SailEnv(wf, polar, p), route_follower(iso, polar, p), s, g)
    dp = SailDP(wf, polar, p, g, device=device)
    st = dp.solve()
    de = rollout(SailEnv(wf, polar, p), dp.policy(), s, g)

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    im = wind_background(ax, wf, p)
    kt_colorbar(fig, im, ax, p)
    for i, fr in enumerate(iso["fronts"]):
        if i % 25 == 0 and i > 0:
            order = np.argsort(np.arctan2(fr[:, 1] - s[1], fr[:, 0] - s[0]))
            ax.plot(fr[order, 0], fr[order, 1], color="white", lw=0.8, alpha=0.55)
    ax.plot(ie["traj"][:, 0], ie["traj"][:, 1], color="#ffd166", lw=2.4,
            label=f"isochrone  {ie['t']:.1f} h, {ie['tacks']} tacks ({iso['compute_s']:.2f} s)")
    ax.plot(de["traj"][:, 0], de["traj"][:, 1], color="#00f5d4", lw=2.4, ls="--",
            label=f"DP  {de['t']:.1f} h, {de['tacks']} tacks ({st['precompute_s'] + st['solve_s']:.2f} s)")
    ax.plot(*s, "o", color="white", mec="black", ms=10)
    ax.plot(*g, "*", color="#ffd166", mec="black", ms=20)
    ax.set(xlim=wf.extent[:2], ylim=wf.extent[2:], aspect="equal",
           title="80 nm dead upwind in 12 kt: you cannot sail straight\n"
                 "white = isochrone fronts every 2.5 h")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "beat.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, ie, de


def fig_random(polar, p, seed, device):
    wf = generate_wind_field(seed)
    rng = np.random.default_rng(0)
    for _ in range(seed + 1):              # same start/goal draw as the planner comparison
        s = rng.uniform(0.5, 9.5, 2)
        gg = rng.uniform(0.5, 9.5, 2)
        while np.linalg.norm(gg - s) < 5.0:
            gg = rng.uniform(0.5, 9.5, 2)
    g = gg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iso = isochrone_route(polar, p, s, g, wind=wf, step_h=0.1, n_sectors=720)
    ie = rollout(SailEnv(wf, polar, p), route_follower(iso, polar, p), s, g)
    dp = SailDP(wf, polar, p, g, device=device)
    st = dp.solve()
    de = rollout(SailEnv(wf, polar, p), dp.policy(), s, g)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6))
    ax = axes[0]
    im = wind_background(ax, wf, p)
    kt_colorbar(fig, im, ax, p)
    ax.plot(ie["traj"][:, 0], ie["traj"][:, 1], color="#ffd166", lw=2.4,
            label=f"isochrone {ie['t']:.1f} h ({iso['compute_s']:.1f} s)")
    ax.plot(de["traj"][:, 0], de["traj"][:, 1], color="#00f5d4", lw=2.4, ls="--",
            label=f"DP {de['t']:.1f} h ({st['precompute_s'] + st['solve_s']:.2f} s)")
    ax.plot(*s, "o", color="white", mec="black", ms=10)
    ax.plot(*g, "*", color="#ffd166", mec="black", ms=20)
    ax.set(xlim=wf.extent[:2], ylim=wf.extent[2:], aspect="equal", title=f"Random wind field (seed {seed})")
    ax.legend(loc="lower left", fontsize=9)

    ax = axes[1]
    V = dp.value_map()
    cf = ax.contourf(dp.xs.cpu().numpy(), dp.ys.cpu().numpy(), V.T, levels=30, cmap="magma")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, label="hours to the goal (best heading)")
    ax.plot(de["traj"][:, 0], de["traj"][:, 1], color="#00f5d4", lw=2.2)
    ax.plot(*s, "o", color="white", mec="black", ms=10)
    ax.plot(*g, "*", color="#ffd166", mec="black", ms=20)
    ax.set(aspect="equal", title="DP value function: time-to-go from every point, one solve")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "random_field.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, ie, de


def fig_timing(polar, p):
    import torch
    wf = generate_wind_field(5)
    g = np.array([8.5, 8.5])
    rows = []
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        SailDP(wf, polar, p, g, nx=41, ny=41, device="cuda").solve()   # warm up kernels
    for n in (61, 81, 121, 161, 241):
        for dev in (["cuda"] if has_cuda else []) + (["cpu"] if n <= 121 else []):
            t0 = time.perf_counter()
            SailDP(wf, polar, p, g, nx=n, ny=n, device=dev).solve()
            if dev == "cuda":
                torch.cuda.synchronize()
            rows.append((n, dev, time.perf_counter() - t0))
            print(f"  DP {n}x{n} x36 on {dev}: {rows[-1][2]:.2f} s", flush=True)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for dev, c, lab in (("cuda", "#00a896", "DP on GPU"), ("cpu", "#e76f51", "DP on CPU")):
        pts = [(n * n * 36, t) for n, d, t in rows if d == dev]
        if pts:
            xs, ts = zip(*pts)
            ax.loglog(xs, ts, "o-", color=c, lw=2, label=lab)
    # a learned policy pays per DECISION, not per state: ~600 decisions for a 30 h voyage
    for ms, ls in ((1.0, "--"), (0.2, ":")):
        ax.axhline(600 * ms / 1000, color="gray", ls=ls, lw=1.4,
                   label=f"learned policy, 600 decisions x {ms:g} ms")
    ax.set(xlabel="DP states (nx * ny * headings)", ylabel="seconds per route",
           title="What a route costs to compute (static wind, one goal)")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "timing.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    polar, p = Polar.synthetic(), SailParams()
    print("saved", fig_polar(polar))
    out, ie, de = fig_beat(polar, p, args.device)
    print(f"saved {out}   beat: isochrone {ie['t']:.2f} h / {ie['tacks']} tacks, DP {de['t']:.2f} h / {de['tacks']} tacks")
    out, ie, de = fig_random(polar, p, args.seed, args.device)
    print(f"saved {out}   random field: isochrone {ie['t']:.2f} h, DP {de['t']:.2f} h")
    if args.bench:
        print("benchmark:")
        print("saved", fig_timing(polar, p))


if __name__ == "__main__":
    main()
