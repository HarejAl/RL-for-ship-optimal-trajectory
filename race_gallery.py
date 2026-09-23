"""
Sweep candidate demo cases for `evolving_race.py`, keep every result, render the best ones.

These clones arrive on roughly a third of crossings, so a presentable race has to be found
rather than assumed. This sweeps generated fields x routes, races all three agents through
each, stores the raw rollouts, and scores the cases on what makes the animation worth
watching:

    all three arrive           - a ship that wanders for 500 steps is not a demo
    in the expected order      - the hurried ship earliest, the thrifty one leanest
    by different paths         - if the three routes overlap it is a speed gap, not a choice
    across lively weather      - a flat blue map says nothing
    with a big fuel ratio      - that is the punchline

Every case is written to output/race_runs/<name>.npz whether it scored well or not, so any
figure can be remade later without re-simulating (see `evolving_race.load_runs`), and the
whole sweep is indexed in output/race_gallery.csv.

    python race_gallery.py                                 # sweep, store, render the top 6
    python race_gallery.py --seeds 1 60 --top 8
    python race_gallery.py --no-render                     # just fill the store and the index

Rendering shells out to `evolving_race.py` so the gallery and a hand-run animation go through
exactly the same code path; the command for each rendered case is printed, ready to re-run
with different framing.
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np

from env import ShipEnv
import evolving_race as R
import preferences as P

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
RUNS_DIR = os.path.join(OUTPUT_DIR, "race_runs")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

# Named routes. All are at least two thirds of the 12-unit map width apart, which is what
# makes the voyage long enough for the three priorities to separate.
ROUTES = {
    "ne":   ((0.8, 1.4), (9.0, 8.4)),   # SW -> NE corner, 10.8 units = 90% of the width
    "se":   ((0.8, 8.6), (9.0, 1.6)),   # NW -> SE corner
    "east": ((0.6, 5.0), (9.4, 5.4)),   # due east, 8.8 units = 73%
    "west": ((9.4, 5.4), (0.6, 5.0)),   # due west, the same water the other way
    "sw":   ((9.0, 8.4), (0.8, 1.4)),   # NE -> SW corner
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs=2, default=[1, 40], metavar=("FIRST", "LAST"),
                    help="inclusive range of generated-field seeds to sweep")
    ap.add_argument("--routes", nargs="+", default=list(ROUTES), choices=list(ROUTES))
    ap.add_argument("--prefs", nargs="+", default=P.ORDER, choices=list(P.PREFERENCES))
    ap.add_argument("--tag-prefix", default="pref")
    ap.add_argument("--drift", type=float, nargs=2, default=[6.0, 3.0])
    ap.add_argument("--slices", type=int, default=49, help="must match the renderer: a case can "
                                                           "change outcome between 25 and 49")
    ap.add_argument("--window-h", type=float, default=24.0)
    ap.add_argument("--update-every-min", type=float, default=90.0)
    ap.add_argument("--max-steps", type=int, default=450,
                    help="a race longer than this makes a tedious animation anyway")
    ap.add_argument("--top", type=int, default=6, help="how many cases to render")
    ap.add_argument("--no-render", action="store_true", help="sweep and store only")
    ap.add_argument("--dpi", type=int, default=76)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--map-every", type=int, default=4)
    return ap.parse_args()


def score_case(runs, prefs, wind_mean):
    """Rank a case by how well it tells the story. Returns (score, metrics dict)."""
    ok = all(runs[n]["success"] for n in prefs)
    t = {n: runs[n]["t_model"] for n in prefs}
    e = {n: runs[n]["fuel"][-1] for n in prefs}
    t_ok = t["fast"] < t["balanced"] < t["eco"]
    e_ok = e["fast"] > e["balanced"] > e["eco"]
    sp_mean, sp_max = R.route_spread(runs) if ok else (0.0, 0.0)
    ratio = e["fast"] / max(e["eco"], 1e-9)
    longest = max(runs[n]["steps"] for n in prefs)
    m = dict(all_arrive=int(ok), time_ordered=int(t_ok), energy_ordered=int(e_ok),
             spread_mean=sp_mean, spread_max=sp_max, fuel_ratio=ratio,
             time_ratio=t["eco"] / max(t["fast"], 1e-9), longest_steps=longest,
             wind_mean=wind_mean)
    if not ok:
        return -1.0, m
    score = (2.0 * t_ok + 2.0 * e_ok + sp_mean + 0.25 * wind_mean
             + 0.6 * np.log(max(ratio, 1.0)) - 0.004 * max(longest - 200, 0))
    return float(score), m


def contact_sheet(names, out_path, ncol=4):
    """
    One panel per candidate showing the finished race, built from the stored runs alone.

    Nothing is re-simulated: the trajectories come out of output/race_runs/*.npz and the
    weather is rebuilt from the `meta` each file carries. This is the sheet to flip through
    when choosing which case to animate.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(names)
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.6 * nrow),
                             facecolor=R.INK, squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, name in zip(axes.ravel(), names):
        runs, prefs, meta = R.load_runs(os.path.join(RUNS_DIR, f"{name}.npz"))
        field = R.drifting_weather(1.0, seed=meta["field_seed"], drift=tuple(meta["drift"]))
        ax.set_axis_on()
        R._paint(ax, field, meta.get("ms_per_unit", 2.5), step=8)
        R._dress_map(ax, field.extent, np.array(meta["start"]), np.array(meta["goal"]),
                     meta["goal_radius"])
        for p in prefs:
            t = runs[p]["traj"]
            ax.plot(t[:, 0], t[:, 1], color=R.NEON[p], lw=2.4, path_effects=R.GLOW,
                    solid_capstyle="round")
        fuel = {p: runs[p]["fuel"][-1] for p in prefs}
        txt = "\n".join(f"{p[:4]:<4s} {runs[p]['hours']:5.1f} h {100 * fuel[p] / max(fuel.values()):3.0f}%"
                        for p in prefs)
        ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=7.5, va="bottom",
                family="monospace", color="white",
                bbox=dict(fc=R.INK, ec="#2a4a63", alpha=0.8, pad=2.5), zorder=10)
        ax.set_title(f"{name}   fuel x{max(fuel.values()) / max(min(fuel.values()), 1e-9):.1f}",
                     color="white", fontsize=11, pad=6)
    fig.suptitle("Candidate races - pick one and re-render it with evolving_race.py",
                 color="white", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=110, facecolor=R.INK)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    os.makedirs(RUNS_DIR, exist_ok=True)
    prefs = list(args.prefs)
    par = {n: P.params(n) for n in prefs}
    paths = {n: os.path.join(MODEL_DIR, f"{args.tag_prefix}_{n}.zip") for n in prefs}

    seeds = list(range(args.seeds[0], args.seeds[1] + 1))
    cases = [(s, r) for s in seeds for r in args.routes]
    print(f"sweeping {len(cases)} cases ({len(seeds)} fields x {len(args.routes)} routes), "
          f"{args.slices} weather slices, max {args.max_steps} steps\n")

    rows = []
    t0 = time.perf_counter()
    for k, (seed, route) in enumerate(cases, 1):
        start, goal = (np.array(p) for p in ROUTES[route])
        slices = R.build_slices("drift", args.slices, args.window_h, seed, args.drift)
        field0 = slices[0][1]
        goal_radius = ShipEnv(field0).goal_radius
        policies = R.load_agents(paths, prefs, field0, goal, par)
        runs, _, sec_per_tu = R.run_case(policies, start, goal, slices, par, prefs, goal_radius,
                                         args.window_h, args.update_every_min, args.max_steps)

        wind_mean = float(field0.speed.mean())
        score, m = score_case(runs, prefs, wind_mean)
        name = f"s{seed}_{route}"
        meta = dict(scenario="drift", field_seed=seed, route=route, drift=list(args.drift),
                    start=list(map(float, start)), goal=list(map(float, goal)),
                    slices=args.slices, window_h=args.window_h, sec_per_tu=sec_per_tu,
                    update_every_min=args.update_every_min, goal_radius=goal_radius,
                    ms_per_unit=2.5, max_steps=args.max_steps, score=score,
                    cost_weights={n: [par[n].time_w, par[n].ctrl_w] for n in prefs})
        R.save_runs(os.path.join(RUNS_DIR, f"{name}.npz"), runs, prefs, meta)

        row = dict(name=name, seed=seed, route=route, score=round(score, 3), **{
            k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in m.items()})
        for n in prefs:
            row[f"{n}_arrived"] = int(runs[n]["success"])
            row[f"{n}_hours"] = round(runs[n]["hours"], 2)
            row[f"{n}_fuel"] = round(float(runs[n]["fuel"][-1]), 1)
            row[f"{n}_steps"] = runs[n]["steps"]
        rows.append(row)

        flag = "" if score < 0 else ("*** " if (m["time_ordered"] and m["energy_ordered"]) else "  + ")
        print(f"[{k:3d}/{len(cases)}] {name:<10s} score {score:6.2f} {flag}"
              + " ".join(f"{n[:4]}{'ok' if runs[n]['success'] else '--'}"
                         f" {runs[n]['hours']:5.1f}h {runs[n]['fuel'][-1]:6.0f}f" for n in prefs)
              + f"  spread {m['spread_mean']:.2f}  fuel x{m['fuel_ratio']:.1f}"
              + f"   [{(time.perf_counter() - t0) / 60:.1f} min]", flush=True)

    index = os.path.join(OUTPUT_DIR, "race_gallery.csv")
    with open(index, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["score"]))
    print(f"\n{len(rows)} cases stored in {RUNS_DIR}\nindex -> {index}")

    good = [r for r in rows if r["score"] > 0]
    ranked = sorted(good, key=lambda r: -r["score"])
    print(f"\n{len(good)} of {len(rows)} cases have all three ships arriving; "
          f"{sum(1 for r in good if r['time_ordered'] and r['energy_ordered'])} are fully ordered\n")
    for r in ranked[:max(args.top, 10)]:
        print(f"  {r['name']:<10s} score {r['score']:6.2f}  spread {r['spread_mean']:.2f}  "
              f"fuel x{r['fuel_ratio']:.1f}  time x{r['time_ratio']:.1f}  wind {r['wind_mean']:.1f}  "
              f"{'ORDERED' if (r['time_ordered'] and r['energy_ordered']) else ''}")

    if ranked:
        sheet = contact_sheet([r["name"] for r in ranked[:max(args.top, 8)]],
                              os.path.join(OUTPUT_DIR, "race_gallery_contact.png"))
        print(f"\ncontact sheet -> {sheet}")

    if args.no_render or not ranked:
        return
    print(f"\nrendering the top {min(args.top, len(ranked))}:")
    for r in ranked[:args.top]:
        s, g = ROUTES[r["route"]]
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "evolving_race.py"),
               "--field-seed", str(r["seed"]),
               "--start", str(s[0]), str(s[1]), "--goal", str(g[0]), str(g[1]),
               "--drift", str(args.drift[0]), str(args.drift[1]),
               "--slices", str(args.slices), "--window-h", str(args.window_h),
               "--max-steps", str(args.max_steps), "--dpi", str(args.dpi),
               "--stride", str(args.stride), "--map-every", str(args.map_every),
               "--tag", f"_{r['name']}"]
        print("  " + " ".join(cmd[1:]))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
        out = subprocess.run(cmd, capture_output=True, text=True, env=env)
        for line in out.stdout.splitlines():
            if "saved" in line:
                print("    " + line.strip())
        if out.returncode != 0:
            print(f"    FAILED: {out.stderr.strip().splitlines()[-1] if out.stderr else '?'}")


if __name__ == "__main__":
    main()
