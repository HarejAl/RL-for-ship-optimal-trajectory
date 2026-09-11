"""
Time-varying wind: the forecast EVOLVES while the ship is under way.

A real Open-Meteo forecast sequence is fetched once for a region. The physical scales in
`field.meta` fix how model time maps to real time (one model time unit = km_per_unit /
ms_per_unit seconds), so a single crossing of the domain spans several real forecast hours
and the wind genuinely changes during the voyage.

Two ships leave the same port for the same destination through the SAME true evolving wind:

  * "adaptive"  - the wind-aware CNN policy re-reads the current forecast every step and
                  reacts to it. Cost: one network forward pass (~1 ms), no re-planning.
  * "departure plan" - dynamic programming solved ONCE on the forecast available at
                  departure, then followed. It never learns that the weather moved.

The gap between them is the value of re-reading the map, and the whole point of an
amortised policy: adapting is free, re-solving is not.

    python receding_horizon_demo.py --model models/bc_t2.zip
    python receding_horizon_demo.py --start 1 1 --goal 9 9 --hours 14

Outputs: output/receding_horizon.gif and output/receding_horizon_summary.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter

from dynamics import ShipParams, ship_step, stage_cost
from env import ShipEnv
from wind import WindField
from dp_baseline import ValueIterationPlanner
from benchmark_dp import load_model
from wind_obs import wrap_wind_obs
from viz import windy_cmap, windy_norm, wind_scale_ticks

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
STROKE = [pe.withStroke(linewidth=3.0, foreground="black")]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/bc_t2.zip")
    ap.add_argument("--lat", type=float, nargs=2, default=[43.0, 44.2])
    ap.add_argument("--lon", type=float, nargs=2, default=[8.2, 9.8])
    ap.add_argument("--nx", type=int, default=24)
    ap.add_argument("--ny", type=int, default=24)
    ap.add_argument("--hours", type=int, default=14, help="forecast hours to fetch (hourly)")
    ap.add_argument("--forecast-days", type=int, default=3, help="Open-Meteo horizon to request")
    ap.add_argument("--ref-speed", type=float, default=25.0,
                    help="m/s mapped to 10 wind units. 25 means a storm-force 25 m/s saturates the "
                         "ship; lower values model a weaker vessel that cannot beat ordinary weather.")
    ap.add_argument("--start", type=float, nargs=2, default=[1.0, 1.5])
    ap.add_argument("--goal", type=float, nargs=2, default=[8.5, 8.5])
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="multiply the physical model->real time factor (>1 = weather evolves faster)")
    ap.add_argument("--update-every-min", type=float, default=None,
                    help="hand the agent a refreshed map every N minutes of simulated real time "
                         "(it holds that map in between). Default: the agent sees the live field.")
    ap.add_argument("--replan", action="store_true",
                    help="also run a re-planning DP that RE-SOLVES on every refreshed map. This is the "
                         "gold standard for adaptive routing and tells you whether the cost of a weather "
                         "change is recoverable at all, or simply unavoidable.")
    ap.add_argument("--sweep", type=float, nargs="+", default=None, metavar="MIN",
                    help="also sweep these refresh intervals (minutes) and plot cost vs cadence; "
                         "use 0 for 'live' and -1 for 'departure map only, never refreshed'")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--dp-nx", type=int, default=61)
    ap.add_argument("--dp-ny", type=int, default=61)
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


class EvolvingWind:
    """Linear interpolation in time between hourly forecast slices sharing one grid."""

    def __init__(self, slices, sec_per_time_unit):
        self.times = [t for t, _ in slices]
        self.fields = [f for _, f in slices]
        self.x, self.y = self.fields[0].x, self.fields[0].y
        self.meta = dict(self.fields[0].meta)
        self.sec_per_time_unit = sec_per_time_unit
        self._cache = {}

    def real_hours(self, model_time):
        return model_time * self.sec_per_time_unit / 3600.0

    def at(self, model_time):
        """WindField for this model time (cached per drawn frame granularity)."""
        h = self.real_hours(model_time)
        key = round(h, 3)
        if key in self._cache:
            return self._cache[key]
        i = int(np.clip(np.floor(h), 0, len(self.fields) - 2))
        a = float(np.clip(h - i, 0.0, 1.0))
        f0, f1 = self.fields[i], self.fields[i + 1]
        wf = WindField(self.x, self.y, (1 - a) * f0.wx + a * f1.wx,
                       (1 - a) * f0.wy + a * f1.wy, meta=self.meta)
        if len(self._cache) < 4000:
            self._cache[key] = wf
        return wf

    def label(self, model_time):
        h = self.real_hours(model_time)
        i = int(np.clip(np.floor(h), 0, len(self.times) - 1))
        return f"{self.times[i][5:16].replace('T', ' ')}Z  (+{h:4.1f} h)"


def simulate(policy, start, goal, evolving, params, goal_radius, max_steps, refresh_min=None):
    """
    Roll out one ship through the true evolving wind. `policy(state, field)` -> thrust.

    `refresh_min` models an operational forecast feed: the agent is handed a NEW map every
    `refresh_min` minutes of simulated real time and must use that same map until the next
    update, while the ship itself always moves through the continuously evolving true wind.
    None means the agent sees the live field at every step.
    """
    state = np.array([start[0], start[1], 0.0, 0.0], dtype=np.float64)
    traj, acts, J = [state.copy()], [], 0.0
    agent_times = []
    success = oob = False
    xmin, xmax, ymin, ymax = evolving.fields[0].extent
    refresh_tu = None
    if refresh_min:
        refresh_tu = refresh_min * 60.0 / evolving.sec_per_time_unit
    for k in range(max_steps):
        t_model = k * params.dt
        field = evolving.at(t_model)                       # truth, drives the dynamics
        if refresh_tu:                                     # what the agent has been given
            t_seen = np.floor(t_model / refresh_tu) * refresh_tu
        else:
            t_seen = t_model
        agent_times.append(t_seen)
        u = np.clip(np.asarray(policy(state, evolving.at(t_seen)), dtype=np.float64),
                    -params.u_max, params.u_max)
        wx, wy = field(state[0], state[1])
        x1, y1, vx1, vy1 = ship_step(state[0], state[1], state[2], state[3],
                                     u[0], u[1], float(wx), float(wy), params)
        state = np.array([x1, y1, vx1, vy1])
        J += float(stage_cost(u[0], u[1], params))
        traj.append(state.copy())
        acts.append(u)
        if np.hypot(x1 - goal[0], y1 - goal[1]) <= goal_radius:
            success = True
            break
        if not (xmin <= x1 <= xmax and ymin <= y1 <= ymax):
            oob = True
            break
    t_model = len(acts) * params.dt
    return dict(traj=np.array(traj), actions=np.array(acts), J=J, success=success, oob=oob,
                t_model=t_model, hours=evolving.real_hours(t_model), steps=len(acts),
                agent_times=np.array(agent_times), refresh_min=refresh_min)


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params = ShipParams()
    start, goal = np.array(args.start), np.array(args.goal)

    print(f"fetching {args.hours} hourly forecast slices  lat={args.lat} lon={args.lon} ...")
    from staleness_study import fetch_cached
    rname = f"rh_{args.lat[0]:g}_{args.lat[1]:g}_{args.lon[0]:g}_{args.lon[1]:g}"
    slices = fetch_cached(rname, args.lat, args.lon, args)
    meta = slices[0][1].meta
    sec_per_tu = meta["km_per_unit"] * 1000.0 / meta["ms_per_unit"] * args.time_scale
    evolving = EvolvingWind(slices, sec_per_tu)
    print(f"scales: {meta['km_per_unit']} km/unit, {meta['ms_per_unit']} (m/s)/unit "
          f"-> 1 model time unit = {sec_per_tu / 3600:.2f} real hours")
    print(f"forecast window {slices[0][0]} .. {slices[-1][0]}")

    goal_radius = ShipEnv(slices[0][1]).goal_radius

    # --- ship 1: departure plan (DP on the forecast at t=0, then followed blind)
    planner = ValueIterationPlanner(slices[0][1], goal, params=params,
                                    nx=args.dp_nx, ny=args.dp_ny, device=args.device)
    st = planner.solve(verbose=False)
    print(f"departure DP plan solved in {st['time']:.1f}s")
    frozen = simulate(lambda s, f: planner.act(s), start, goal, evolving, params,
                      goal_radius, args.max_steps)
    # control: the same plan under the weather it ASSUMED (t=0 field held constant).
    # If this succeeds while `frozen` fails, the plan was sound and the weather moving broke it.
    still = EvolvingWind([slices[0], slices[0]], sec_per_tu)
    as_intended = simulate(lambda s, f: planner.act(s), start, goal, still, params,
                           goal_radius, args.max_steps)

    # --- ship 2: adaptive policy re-reading the live forecast every step
    model, obs_cfg = load_model(args.model)
    host = ShipEnv(slices[0][1], params=params)
    host.goal = goal.astype(np.float64)
    wrapper = wrap_wind_obs(host, obs_cfg)

    def adaptive_policy(state, field):
        host.wind = field
        host.state = state.astype(np.float64)
        obs = wrapper.observation(None)
        a, _ = model.predict(obs, deterministic=True)
        return a

    adaptive = simulate(adaptive_policy, start, goal, evolving, params, goal_radius,
                        args.max_steps, refresh_min=args.update_every_min)
    if args.update_every_min:
        print(f"agent map refresh: every {args.update_every_min:g} min "
              f"({args.update_every_min * 60 / sec_per_tu:.3f} model time units)")

    print()
    for name, r in (("plan, weather HELD (control)", as_intended),
                    ("plan, weather MOVED", frozen),
                    ("adaptive policy, weather MOVED", adaptive)):
        status = "ARRIVED" if r["success"] else ("left the area" if r["oob"] else "ran out of time")
        print(f"{name:32s} {status:15s} J={r['J']:6.2f}  voyage={r['hours']:5.1f} h  steps={r['steps']}")
    if as_intended["success"] and not frozen["success"]:
        print("\n=> the departure plan was valid for the forecast it was made on; the weather moving broke it.")
    if adaptive["success"] and not frozen["success"]:
        print("=> re-reading the live map (one ~1 ms forward pass per step) completed the voyage.")
    horizon_h = len(slices) - 1
    longest = max(r["hours"] for r in (frozen, adaptive))
    if longest > horizon_h:
        print(f"\nWARNING: voyage reaches {longest:.1f} h but only {horizon_h} forecast hours were "
              f"fetched; the tail used the last slice held constant. Increase --hours.")

    if args.replan:
        solves = {}

        def replanning_policy(state, field):
            """Re-solve DP whenever a new map arrives (fields are cached per refresh, so
            `id` identifies the map the agent currently holds)."""
            key = id(field)
            if key not in solves:
                p = ValueIterationPlanner(field, goal, params=params, nx=args.dp_nx,
                                          ny=args.dp_ny, device=args.device)
                p.solve(verbose=False)
                solves[key] = p
            return solves[key].act(state)

        replan = simulate(replanning_policy, start, goal, evolving, params, goal_radius,
                          args.max_steps, refresh_min=args.update_every_min)
        print(f"{'re-planning DP (re-solve each map)':32s} "
              f"{'ARRIVED' if replan['success'] else 'failed':15s} J={replan['J']:6.2f}  "
              f"voyage={replan['hours']:5.1f} h  ({len(solves)} DP solves)")
        base = as_intended["J"]
        print(f"    vs departure optimum {(replan['J'] / base - 1) * 100:+.1f}%   "
              f"| stale plan {(frozen['J'] / base - 1) * 100:+.1f}%   "
              f"| learned policy {(adaptive['J'] / base - 1) * 100:+.1f}%")
        if frozen["success"] and replan["success"]:
            gain = (frozen["J"] - replan["J"]) / frozen["J"] * 100
            print(f"    => re-planning recovers {gain:+.1f}% of the stale plan's cost "
                  f"({'worth it' if gain > 3 else 'essentially nothing - the extra cost is unavoidable'})")

    if args.sweep:
        refresh_sweep(adaptive_policy, start, goal, evolving, params, goal_radius,
                      args, as_intended["J"])

    make_animation(evolving, start, goal, goal_radius, frozen, adaptive, params, args)
    make_summary(evolving, start, goal, goal_radius, frozen, adaptive, as_intended, args)


def refresh_sweep(policy, start, goal, evolving, params, goal_radius, args, ref_J):
    """How often does the map actually need refreshing? Same voyage, different cadences."""
    print("\n--- refresh cadence sweep ---")
    labels, costs, oks = [], [], []
    huge = evolving.real_hours(1e6) * 60  # an interval longer than any voyage = never refresh
    for m in args.sweep:
        if m == 0:
            lab, refresh = "live", None
        elif m < 0:
            lab, refresh = "never", huge
        else:
            lab, refresh = (f"{m / 60:g} h" if m >= 60 else f"{m:g} min"), m
        r = simulate(policy, start, goal, evolving, params, goal_radius, args.max_steps,
                     refresh_min=refresh)
        labels.append(lab)
        costs.append(r["J"])
        oks.append(r["success"])
        print(f"  refresh {lab:>8s}: {'ARRIVED' if r['success'] else 'FAILED ':8s} "
              f"J={r['J']:6.2f}  ({(r['J'] / ref_J - 1) * 100:+5.1f}% vs the departure optimum)  "
              f"voyage={r['hours']:4.1f} h", flush=True)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs = np.arange(len(labels))
    cols = ["#2a9d8f" if ok else "#c1121f" for ok in oks]
    ax.bar(xs, costs, color=cols)
    ax.axhline(ref_J, color="black", ls="--", lw=1.2, label="departure-forecast optimum")
    for i, (c, ok) in enumerate(zip(costs, oks)):
        ax.text(i, c, f"{c:.2f}" + ("" if ok else "\nFAILED"), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set(xlabel="how often the agent is handed a new map", ylabel="voyage cost J",
           title="Does the refresh cadence matter?")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "refresh_sweep.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def _draw_field(ax, field, quiver_step=2, ms_per_unit=1.0):
    norm = windy_norm(ms_per_unit)
    im = ax.pcolormesh(field.x, field.y, field.speed.T, shading="gouraud",
                       cmap=windy_cmap(), norm=norm)
    X, Y = np.meshgrid(field.x, field.y, indexing="ij")
    s = quiver_step
    q = ax.quiver(X[::s, ::s], Y[::s, ::s], field.wx[::s, ::s], field.wy[::s, ::s],
                  color="white", scale=150, width=0.003, alpha=0.85)
    return im, q


def make_animation(evolving, start, goal, goal_radius, frozen, adaptive, params, args):
    n = max(len(frozen["traj"]), len(adaptive["traj"]))
    frames = (n + args.stride - 1) // args.stride
    f0 = evolving.at(0.0)

    msu = evolving.meta.get("ms_per_unit", 1.0)
    fig, ax = plt.subplots(figsize=(8.0, 7.4))
    im, q = _draw_field(ax, f0, ms_per_unit=msu)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks, labels = wind_scale_ticks(msu)
    cb.set_ticks(ticks); cb.set_ticklabels(labels)
    cb.set_label("wind speed (m/s)")
    ax.plot(*start, "o", color="white", mec="black", ms=9, zorder=5)
    ax.plot(*goal, "s", color="gold", mec="black", ms=11, zorder=5, label="destination")
    ax.add_patch(plt.Circle(goal, goal_radius, fill=False, ec="gold", lw=1.2, ls=":"))

    (l_f,) = ax.plot([], [], color="#ff5555", lw=2.6, ls="--", label="departure plan (DP @ t=0)",
                     zorder=6, path_effects=STROKE)
    (l_a,) = ax.plot([], [], color="cyan", lw=2.6, label="adaptive policy (live map)",
                     zorder=6, path_effects=STROKE)
    (d_f,) = ax.plot([], [], "o", color="#ff5555", mec="black", mew=1.3, ms=11, zorder=7)
    (d_a,) = ax.plot([], [], "o", color="cyan", mec="black", mew=1.3, ms=11, zorder=7)
    xmin, xmax, ymin, ymax = f0.extent
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x", ylabel="y")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    clock = ax.text(0.5, 1.012, "", transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=12, family="monospace", weight="bold")
    txt = ax.text(0.985, 0.015, "", transform=ax.transAxes, ha="right", va="bottom",
                  fontsize=9.5, family="monospace",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.88))
    ax.set_title("Forecast evolves during the voyage" +
                 (f" - agent re-fed the map every {args.update_every_min:g} min"
                  if args.update_every_min else ""), fontsize=13, pad=26)

    def cum(actions):
        c = params.dt * (params.time_w + params.ctrl_w * (actions ** 2).sum(axis=1))
        return np.concatenate(([0.0], np.cumsum(c)))
    Jf, Ja = cum(frozen["actions"]), cum(adaptive["actions"])

    at = adaptive.get("agent_times")
    refresh = adaptive.get("refresh_min")

    def update(k):
        i = k * args.stride
        t_model = i * params.dt
        # show the map the agent is actually holding (it only refreshes every N minutes)
        t_shown = float(at[min(i, len(at) - 1)]) if at is not None and len(at) else t_model
        field = evolving.at(t_shown)
        im.set_array(field.speed.T.ravel())
        q.set_UVC(field.wx[::2, ::2], field.wy[::2, ::2])
        out = []
        for res, line, dot, Jc in ((frozen, l_f, d_f, Jf), (adaptive, l_a, d_a, Ja)):
            j = min(i, len(res["traj"]) - 1)
            line.set_data(res["traj"][:j + 1, 0], res["traj"][:j + 1, 1])
            dot.set_data([res["traj"][j, 0]], [res["traj"][j, 1]])
            out.append((Jc[min(j, len(Jc) - 1)], j >= len(res["traj"]) - 1 and res["success"]))
        lab = evolving.label(t_model)
        if refresh:
            lab += f"   |  map refreshed every {refresh:g} min  (showing {evolving.label(t_shown)[:11]})"
        clock.set_text(lab)
        txt.set_text(f"departure plan  J={out[0][0]:5.2f}  {'ARRIVED' if out[0][1] else ''}\n"
                     f"adaptive policy J={out[1][0]:5.2f}  {'ARRIVED' if out[1][1] else ''}")
        return [im, q, l_f, l_a, d_f, d_a, clock, txt]

    fig.tight_layout()
    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    out = os.path.join(OUTPUT_DIR, "receding_horizon.gif")
    ani.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"saved {out}")


def make_summary(evolving, start, goal, goal_radius, frozen, adaptive, as_intended, args):
    """Three snapshots of the evolving forecast with both tracks drawn up to that moment."""
    n = max(len(frozen["traj"]), len(adaptive["traj"]))
    picks = [0, n // 2, n - 1]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    p = ShipParams()
    for pi, (ax, i) in enumerate(zip(axes, picks)):
        t_model = i * p.dt
        field = evolving.at(t_model)
        im, _ = _draw_field(ax, field, quiver_step=2, ms_per_unit=evolving.meta.get("ms_per_unit", 1.0))
        if pi == 0:  # what the departure plan expected to do, had the weather held
            ax.plot(as_intended["traj"][:, 0], as_intended["traj"][:, 1], color="white", lw=1.6,
                    ls=(0, (4, 3)), alpha=0.95, label="plan as intended (if weather held)")
            ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
        for res, c, ls in ((frozen, "#ff5555", "--"), (adaptive, "cyan", "-")):
            j = min(i, len(res["traj"]) - 1)
            ax.plot(res["traj"][:j + 1, 0], res["traj"][:j + 1, 1], color=c, ls=ls, lw=2.4,
                    path_effects=STROKE)
            ax.plot(res["traj"][j, 0], res["traj"][j, 1], "o", color=c, mec="black", ms=10)
        ax.plot(*start, "o", color="white", mec="black", ms=8)
        ax.plot(*goal, "s", color="gold", mec="black", ms=10)
        ax.set(title=evolving.label(t_model), xlabel="x", ylabel="y")
        ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=axes, fraction=0.02)
    ticks, labels = wind_scale_ticks(evolving.meta.get("ms_per_unit", 1.0))
    cb.set_ticks(ticks); cb.set_ticklabels(labels); cb.set_label("wind speed (m/s)")
    refresh = adaptive.get("refresh_min")
    seen = f"re-fed the map every {refresh:g} min" if refresh else "re-reading the live map"
    outcome = []
    for lab, r in (("departure plan", frozen), ("policy", adaptive)):
        outcome.append(f"{lab} {'arrived' if r['success'] else ('left the area' if r['oob'] else 'timed out')}"
                       f" J={r['J']:.2f}")
    fig.suptitle("Same voyage, weather moving underneath it:  red = plan fixed at departure,  "
                 f"cyan = policy {seen}\n" + "   |   ".join(outcome), fontsize=12)
    out = os.path.join(OUTPUT_DIR, "receding_horizon_summary.png")
    fig.savefig(out, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
