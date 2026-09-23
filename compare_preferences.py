"""
Compare the preference-specific agents (fast / balanced / eco) on identical held-out cases.

Every agent is rolled out on the same benchmark cases as `benchmark_dp.py` (a fresh
generated wind field per case, start and goal drawn by the seeded env), and each case is
also solved by the DP planner under each of the three cost weightings, which gives the
optimal reference for every objective.

Because the stage cost is l(u) = dt * (time_w + ctrl_w |u|^2), the cost of a trajectory
under ANY preference is an affine function of just two numbers,

    J_p = time_w(p) * T + ctrl_w(p) * E,      T = arrival time, E = sum |u|^2 dt,

so one rollout per agent is enough to score it under all three objectives. That is what
makes the cross-cost matrix below cheap: it is not three evaluations, it is one rollout
re-priced three ways.

    python compare_preferences.py --n-cases 30 --tag-prefix pref

Outputs
    output/preference_benchmark.csv      one row per (case, policy)
    output/preference_pareto.png/.pdf    time-energy Pareto plot
    output/preference_trajectories.*     routes of the three agents on example cases
    printed: success rates, medians, optimality gap vs each agent's own DP, cross-cost matrix
"""

import argparse
import csv
import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from env import ShipEnv
from wind import generate_wind_field
from dp_baseline import ValueIterationPlanner
from benchmark_dp import load_model
import preferences as P

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
STYLE = os.path.join(SCRIPT_DIR, "journal.mplstyle")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefs", nargs="+", default=P.ORDER, choices=list(P.PREFERENCES))
    ap.add_argument("--tag-prefix", default="pref",
                    help="agent for preference <p> is models/<tag-prefix>_<p>.zip")
    ap.add_argument("--models", nargs="*", default=None, metavar="PREF=PATH",
                    help="explicit model paths, overriding --tag-prefix")
    ap.add_argument("--n-cases", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0, help="case seeds are seed*10000 + k, as in benchmark_dp.py")
    ap.add_argument("--no-dp", action="store_true", help="skip the DP references (agents only)")
    ap.add_argument("--nx", type=int, default=61)
    ap.add_argument("--ny", type=int, default=61)
    ap.add_argument("--nv", type=int, default=13)
    ap.add_argument("--n-act", type=int, default=5)
    ap.add_argument("--exec-n-act", type=int, default=9)
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--traj-cases", type=int, nargs="*", default=None,
                    help="case indices to draw in the trajectory figure (default: the first 3 solved by all)")
    ap.add_argument("--tag", default="")
    return ap.parse_args()


# --------------------------------------------------------------------- rollouts
def rollout_policy(model, env, start, goal, wind):
    """Roll out an SB3 policy and return the trajectory plus the (T, E) summary."""
    base = env.unwrapped
    obs, info = env.reset(options=dict(start=start, goal=goal, wind=wind))
    t0 = time.perf_counter()
    traj, actions = [base.state.copy()], []
    for _ in range(base.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(a)
        traj.append(base.state.copy())
        actions.append(np.clip(np.asarray(a, dtype=np.float64), -base.p.u_max, base.p.u_max))
        if terminated or truncated:
            break
    return summarize(np.array(traj), np.array(actions), info, base.p, time.perf_counter() - t0)


def summarize(traj, actions, info, p, wall):
    """(T, E) summary of a rollout; costs under any preference follow from these two."""
    return dict(traj=traj, actions=actions, success=bool(info["success"]), oob=bool(info["oob"]),
                T=float(info["t"]), E=P.energy(actions, p), steps=len(actions), wall=wall)


def cost(res, name):
    """Cost of a rollout under preference `name`: J = time_w * T + ctrl_w * E."""
    time_w, ctrl_w, _ = P.PREFERENCES[name]
    return time_w * res["T"] + ctrl_w * res["E"]


# ------------------------------------------------------------------- reporting
def pct(x):
    return "n/a" if not np.isfinite(x) else f"{x:+.1f}%"


def median_gap(rows, key_num, key_den):
    g = [100 * (r[key_num] / r[key_den] - 1) for r in rows if np.isfinite(r[key_num]) and r[key_den] > 0]
    return (np.median(g), np.mean(g), len(g)) if g else (np.nan, np.nan, 0)


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prefs = list(args.prefs)

    paths = {n: os.path.join(MODEL_DIR, f"{args.tag_prefix}_{n}.zip") for n in prefs}
    for spec in (args.models or []):
        n, _, path = spec.partition("=")
        paths[n] = path
    agents = {}
    for n in prefs:
        model, obs_cfg = load_model(paths[n])
        agents[n] = (model, obs_cfg)
        print(f"agent {n:<9s} <- {paths[n]}  (wind-aware={obs_cfg is not None})")
    print()
    for n in prefs:
        print("  " + P.describe(n))
    print()

    par = {n: P.params(n) for n in prefs}
    records = []          # flat rows for the CSV
    cases = []            # per-case dict of results, for the figures

    for k in range(args.n_cases):
        case_seed = args.seed * 10_000 + k
        wind = generate_wind_field(case_seed)
        env = ShipEnv(wind)
        env.reset(seed=case_seed)
        start, goal = env.state[:2].copy(), env.goal.copy()
        case = dict(k=k, seed=case_seed, wind=wind, start=start, goal=goal, dp={}, rl={})

        # --- DP optimum for each objective (one precompute, three value iterations)
        if not args.no_dp:
            planner = ValueIterationPlanner(wind, goal, params=par[prefs[0]], nx=args.nx, ny=args.ny,
                                            nv=args.nv, n_act=args.n_act, exec_n_act=args.exec_n_act,
                                            device=args.device)
            for n in prefs:
                planner.set_cost_weights(par[n])
                env.p = par[n]
                st = planner.solve(max_iter=args.max_iter, verbose=False)
                t0 = time.perf_counter()
                res = planner.rollout(env, start)
                res = summarize(res["traj"], res["actions"],
                                dict(success=res["success"], oob=res["oob"], t=res["t"]),
                                par[n], time.perf_counter() - t0)
                res["solve"] = st["time"]
                case["dp"][n] = res

        # --- the three agents on the same case
        for n in prefs:
            model, obs_cfg = agents[n]
            env.p = par[n]
            rl_env = env
            if obs_cfg is not None:
                from wind_obs import wrap_wind_obs
                rl_env = wrap_wind_obs(env, obs_cfg)
            case["rl"][n] = rollout_policy(model, rl_env, start, goal, wind)

        # --- flatten to CSV rows
        for kind, block in (("dp", case["dp"]), ("agent", case["rl"])):
            for n, res in block.items():
                row = dict(case=k, seed=case_seed, kind=kind, pref=n, success=int(res["success"]),
                           oob=int(res["oob"]), T=res["T"], E=res["E"], steps=res["steps"],
                           wall=res["wall"], solve=res.get("solve", 0.0))
                for m in prefs:
                    row[f"J_{m}"] = cost(res, m)
                records.append(row)
        cases.append(case)

        line = f"case {k:3d} |"
        for n in prefs:
            r = case["rl"][n]
            tag = "ok " if r["success"] else ("oob" if r["oob"] else "t/o")
            line += f" {n[:4]} {tag} T={r['T']:5.2f} E={r['E']:7.1f}"
            if not args.no_dp:
                d = case["dp"][n]
                gap = 100 * (cost(r, n) / cost(d, n) - 1) if (r["success"] and d["success"]) else np.nan
                line += f" ({pct(gap)} vs DP)"
            line += " |"
        print(line, flush=True)

    # ------------------------------------------------------------------ CSV
    csv_path = os.path.join(OUTPUT_DIR, f"preference_benchmark{args.tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"\nper-case results -> {csv_path}")

    # --------------------------------------------------------------- summary
    def rows(kind, n):
        return [r for r in records if r["kind"] == kind and r["pref"] == n]

    print("\n" + "=" * 96)
    print(f"{'agent':<10s} {'success':>8s} {'median T':>10s} {'median E':>10s} "
          f"{'median J':>10s} {'gap vs own DP':>16s} {'ms online':>11s}")
    print(f"{'':<10s} {'':>8s} {'':>10s} {'':>10s} {'':>10s} {'':>16s} "
          f"{'(agent: rollout, DP: solve)':>11s}")
    print("-" * 96)
    for n in prefs:
        rl = rows("agent", n)
        ok = [r for r in rl if r["success"]]
        med_T = np.median([r["T"] for r in ok]) if ok else np.nan
        med_E = np.median([r["E"] for r in ok]) if ok else np.nan
        med_J = np.median([r[f"J_{n}"] for r in ok]) if ok else np.nan
        gap_txt = "n/a"
        if not args.no_dp:
            dp = {r["case"]: r for r in rows("dp", n)}
            pairs = [dict(num=r[f"J_{n}"], den=dp[r["case"]][f"J_{n}"])
                     for r in ok if dp.get(r["case"], {}).get("success")]
            med, mean, cnt = median_gap(pairs, "num", "den")
            gap_txt = f"{pct(med)} ({cnt} cases)"
        print(f"{n:<10s} {100 * len(ok) / max(len(rl), 1):7.0f}% {med_T:10.2f} {med_E:10.1f} "
              f"{med_J:10.2f} {gap_txt:>16s} {1e3 * np.mean([r['wall'] for r in ok]) if ok else np.nan:10.0f}")
    if not args.no_dp:
        print("-" * 96)
        for n in prefs:
            dp = [r for r in rows("dp", n) if r["success"]]
            print(f"{'DP ' + n:<10s} {100 * len(dp) / max(args.n_cases, 1):7.0f}% "
                  f"{np.median([r['T'] for r in dp]):10.2f} {np.median([r['E'] for r in dp]):10.1f} "
                  f"{np.median([r[f'J_{n}'] for r in dp]):10.2f} {'reference':>16s} "
                  f"{1e3 * np.mean([r['solve'] for r in dp]):10.0f}")
    print("=" * 96)

    # cross-cost matrix: agent i scored under objective j, relative to the best agent in that column
    common = [c for c in cases if all(c["rl"][n]["success"] for n in prefs)]
    print(f"\ncross-cost matrix: mean J of each agent under each objective, over the "
          f"{len(common)} cases all agents solved")
    print("(each column should be smallest on the diagonal: the agent trained for that "
          "objective wins it)\n")
    head = f"{'agent \\ cost':<14s}" + "".join(f"{'J_' + m:>12s}" for m in prefs) + f"{'T':>9s}{'E':>10s}"
    print(head)
    print("-" * len(head))
    M = np.zeros((len(prefs), len(prefs)))
    for i, n in enumerate(prefs):
        Js = [np.mean([cost(c["rl"][n], m) for c in common]) for m in prefs]
        M[i] = Js
        T = np.mean([c["rl"][n]["T"] for c in common])
        E = np.mean([c["rl"][n]["E"] for c in common])
        print(f"{n:<14s}" + "".join(f"{v:12.2f}" for v in Js) + f"{T:9.2f}{E:10.1f}")
    print("-" * len(head))
    win = [prefs[int(np.argmin(M[:, j]))] for j in range(len(prefs))]
    print(f"{'best agent':<14s}" + "".join(f"{w:>12s}" for w in win))
    print("diagonal wins: " + ("YES, every objective is won by its own agent"
                               if all(w == m for w, m in zip(win, prefs))
                               else "NO -> " + ", ".join(f"{m}: {w}" for m, w in zip(prefs, win))))

    # ------------------------------------------------------------------ plots
    plt.style.use(STYLE)
    mm = 1 / 25.4

    # Pareto: arrival time vs control energy, one marker per (case, agent)
    fig, ax = plt.subplots(figsize=(85 * mm, 75 * mm), layout="constrained")
    for c in common:
        ax.plot([c["rl"][n]["T"] for n in prefs], [c["rl"][n]["E"] for n in prefs],
                "-", color="0.8", lw=0.5, zorder=1)
    for n in prefs:
        T = [c["rl"][n]["T"] for c in common]
        E = [c["rl"][n]["E"] for c in common]
        ax.plot(T, E, "o", color=P.COLORS[n], ms=3.5, alpha=0.65, zorder=2, label=f"{n} agent")
        ax.plot(np.mean(T), np.mean(E), "*", color=P.COLORS[n], ms=13, zorder=4,
                markeredgecolor="k", markeredgewidth=0.4)
    if not args.no_dp:
        for n in prefs:
            d = [c["dp"][n] for c in common if c["dp"][n]["success"]]
            if d:
                ax.plot(np.mean([r["T"] for r in d]), np.mean([r["E"] for r in d]), "s",
                        ms=7, zorder=3, markeredgecolor=P.COLORS[n], markeredgewidth=1.4,
                        markerfacecolor="none")
    ax.set(xlabel="arrival time $T$", ylabel=r"control energy $E=\sum|u|^2\,\mathrm{d}t$", yscale="log")
    ax.spines[["top", "right"]].set_visible(False)
    extra = [plt.Line2D([], [], ls="none", marker="*", ms=10, color="0.3", label="agent mean"),
             plt.Line2D([], [], ls="none", marker="s", ms=7, markerfacecolor="none",
                        markeredgecolor="0.3", label="DP optimum")]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + extra, frameon=False, loc="upper right",
              fontsize=7, handletextpad=0.4, labelspacing=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"preference_pareto{args.tag}.{ext}"))

    # Trajectories on a few example cases
    pick = args.traj_cases or [c["k"] for c in common[:3]]
    pick = [k for k in pick if k < len(cases)]
    if pick:
        fig, axes = plt.subplots(1, len(pick), figsize=(min(170, 58 * len(pick)) * mm, 68 * mm),
                                 layout="constrained", squeeze=False)
        handles = None
        for ax, k in zip(axes.ravel(), pick):
            c = cases[k]
            w = c["wind"]
            # keep the wind light so the coloured routes stay readable
            ax.pcolormesh(w.x, w.y, w.speed.T, shading="auto", cmap="Greys",
                          vmin=0.0, vmax=1.7 * float(w.speed.max()))
            s = 8
            X, Y = np.meshgrid(w.x, w.y, indexing="ij")
            ax.quiver(X[::s, ::s], Y[::s, ::s], w.wx[::s, ::s], w.wy[::s, ::s],
                      color="0.45", scale=170, width=0.004)
            ax.set_aspect("equal")
            txt = []
            for n in prefs:
                r = c["rl"][n]
                ax.plot(r["traj"][:, 0], r["traj"][:, 1], "-", color=P.COLORS[n], lw=1.7, zorder=3)
                if not args.no_dp and c["dp"].get(n, {}).get("success"):
                    d = c["dp"][n]
                    ax.plot(d["traj"][:, 0], d["traj"][:, 1], ":", color=P.COLORS[n], lw=1.1, zorder=2)
                txt.append(f"{n[:4]}  T={r['T']:.2f}  E={r['E']:.0f}")
            ax.plot(*c["start"], "o", color="k", ms=4, zorder=4)
            ax.plot(*c["goal"], "*", color="k", ms=10, zorder=4)
            # park the T/E box in whichever half of the panel the routes leave empty
            ys = np.concatenate([c["rl"][n]["traj"][:, 1] for n in prefs])
            lo, hi = ax.get_ylim()
            top = np.mean((ys - lo) / (hi - lo)) < 0.5
            ax.text(0.03, 0.97 if top else 0.03, "\n".join(txt), transform=ax.transAxes, fontsize=6,
                    va="top" if top else "bottom", family="monospace",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5), zorder=5)
            ax.set(title=f"case {k}", xlabel="$x$")
            if handles is None:
                handles = [plt.Line2D([], [], color=P.COLORS[n], lw=1.7, label=f"{n} agent") for n in prefs]
                if not args.no_dp:
                    handles.append(plt.Line2D([], [], color="0.35", lw=1.1, ls=":", label="DP optimum"))
        axes.ravel()[0].set_ylabel("$y$")
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=8)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUTPUT_DIR, f"preference_trajectories{args.tag}.{ext}"))

    print(f"\nfigures -> {os.path.join(OUTPUT_DIR, 'preference_pareto' + args.tag + '.png')}, "
          f"{os.path.join(OUTPUT_DIR, 'preference_trajectories' + args.tag + '.png')}")
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg"):
        plt.show()


if __name__ == "__main__":
    main()
