"""
Build one DP-teacher dataset per cost preference, from a single pass over the fields.

This is the data stage of the multi-objective study: the same wind fields, the same
goals, the same rollout starts and the same random states are labelled by three DP
teachers that differ only in the stage-cost weights (see `preferences.py`). Anything
the resulting agents do differently therefore comes from the objective, not from the
data draw.

Value iteration is re-solved per preference - the value function is the fixed point of
its own cost - but the expensive part of the setup (successor cells and interpolation
weights of every state-action pair, which depend on the dynamics only) is built once
per (field, goal) and reused, via `ValueIterationPlanner.set_cost_weights`.

    python dp_dataset_prefs.py --n-fields 130 --goals-per-field 2 --rollouts 20

Output: data/<prefix>_<pref>.npz, same columns as `dp_dataset.py`
    field_seed (N,)  goal (N,2)  state (N,4)  action (N,2)  value (N,)  source (N,)
    next_state (N,4)  reward (N,)  terminated (N,)      (source 0 = DP rollout, 1 = random)
Each file is rewritten after every field, so a partial run is usable and the run can be
resumed with --first-field.
"""

import argparse
import os
import time

import numpy as np

from env import ShipEnv
from wind import generate_wind_field
from dp_baseline import ValueIterationPlanner
from wind_obs import TRAIN_SEED_BASE
import preferences as P

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

COLS = ("field_seed", "goal", "state", "action", "value", "source",
        "next_state", "reward", "terminated")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefs", nargs="+", default=P.ORDER, choices=list(P.PREFERENCES),
                    help="cost-weight presets to build a teacher for")
    ap.add_argument("--n-fields", type=int, default=130)
    ap.add_argument("--first-field", type=int, default=0, help="index of the first training field")
    ap.add_argument("--goals-per-field", type=int, default=2)
    ap.add_argument("--rollouts", type=int, default=20, help="DP rollouts from random starts per (field, goal)")
    ap.add_argument("--random-states", type=int, default=1500, help="random labelled states per (field, goal)")
    ap.add_argument("--vel-std", type=float, default=1.5, help="std of random-state velocities")
    ap.add_argument("--nx", type=int, default=61)
    ap.add_argument("--ny", type=int, default=61)
    ap.add_argument("--nv", type=int, default=13)
    ap.add_argument("--n-act", type=int, default=5)
    ap.add_argument("--exec-n-act", type=int, default=9)
    ap.add_argument("--goal-bonus", type=float, default=10.0)
    ap.add_argument("--oob-penalty", type=float, default=10.0)
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--prefix", default="dp_teacher_pref",
                    help="output files are data/<prefix>_<pref>.npz")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    prefs = list(args.prefs)
    par = {n: P.params(n) for n in prefs}
    out_path = {n: os.path.join(DATA_DIR, f"{args.prefix}_{n}.npz") for n in prefs}
    cols = {n: {k: [] for k in COLS} for n in prefs}
    stats = {n: dict(solves=0, vi_time=0.0, ok=0, rollouts=0) for n in prefs}

    print("preferences:")
    for n in prefs:
        print(f"  {P.describe(n):<95s} -> {out_path[n]}")

    rng = np.random.default_rng(args.seed)
    t_start = time.perf_counter()

    for i in range(args.first_field, args.first_field + args.n_fields):
        seed = TRAIN_SEED_BASE + i
        wind = generate_wind_field(seed)
        env = ShipEnv(wind, goal_bonus=args.goal_bonus, oob_penalty=args.oob_penalty)
        xmin, xmax, ymin, ymax = wind.extent

        for g in range(args.goals_per_field):
            # --- draw the case ONCE; every preference sees the same goal, starts and states
            goal = rng.uniform(*env.spawn_box, size=2)
            starts = []
            while len(starts) < args.rollouts:
                s = rng.uniform(*env.spawn_box, size=2)
                if np.linalg.norm(s - goal) >= env.min_start_goal_dist:
                    starts.append(s)
            n = args.random_states
            S = np.empty((n, 4))
            S[:, 0] = rng.uniform(xmin, xmax, n)
            S[:, 1] = rng.uniform(ymin, ymax, n)
            S[:, 2:] = rng.normal(0.0, args.vel_std, (n, 2))

            planner = ValueIterationPlanner(wind, goal, params=par[prefs[0]], nx=args.nx, ny=args.ny,
                                            nv=args.nv, n_act=args.n_act, exec_n_act=args.exec_n_act,
                                            device=args.device)
            S[:, 2:] = np.clip(S[:, 2:], -planner.v_max, planner.v_max)

            line = []
            for name in prefs:
                planner.set_cost_weights(par[name])
                env.p = par[name]
                st = planner.solve(max_iter=args.max_iter, verbose=False)
                stats[name]["solves"] += 1
                stats[name]["vi_time"] += st["time"]
                c = cols[name]

                # --- DP rollouts from the shared starts: states, greedy actions, real transitions
                n_ok = 0
                for start in starts:
                    res = planner.rollout(env, start)
                    n_ok += int(res["success"])
                    T = res["steps"]
                    traj = res["traj"][:-1]
                    c["field_seed"].append(np.full(T, seed))
                    c["goal"].append(np.tile(goal, (T, 1)))
                    c["state"].append(traj)
                    c["action"].append(res["actions"])
                    c["value"].append(planner.value(traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]))
                    c["source"].append(np.zeros(T, dtype=np.int8))
                    c["next_state"].append(res["traj"][1:])
                    c["reward"].append(res["rewards"])
                    term = np.zeros(T, dtype=bool)
                    term[-1] = res["terminated"]
                    c["terminated"].append(term)
                stats[name]["ok"] += n_ok
                stats[name]["rollouts"] += len(starts)

                # --- the shared random states, labelled with this teacher's greedy action
                A, _ = planner.act_batch(S)
                c["field_seed"].append(np.full(n, seed))
                c["goal"].append(np.tile(goal, (n, 1)))
                c["state"].append(S)
                c["action"].append(A)
                c["value"].append(planner.value(S[:, 0], S[:, 1], S[:, 2], S[:, 3]))
                c["source"].append(np.ones(n, dtype=np.int8))
                c["next_state"].append(np.full((n, 4), np.nan))
                c["reward"].append(np.zeros(n))
                c["terminated"].append(np.zeros(n, dtype=bool))

                line.append(f"{name} {st['time']:4.1f}s/{st['iterations']:4d}it ok {n_ok:2d}/{len(starts)}")

            print(f"field {i:3d} goal {g} | " + " | ".join(line) +
                  f" | rows {sum(len(a) for a in cols[prefs[0]]['state']):,}"
                  f" | elapsed {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)

        # rewrite after each field so partial runs are usable
        for name in prefs:
            s = stats[name]
            np.savez_compressed(out_path[name],
                                **{k: np.concatenate(v) for k, v in cols[name].items()},
                                meta=np.array([s["solves"], s["vi_time"], args.nx, args.ny, args.nv]),
                                cost_weights=np.array([par[name].time_w, par[name].ctrl_w]),
                                pref=np.array(name))

    print(f"\ndone in {(time.perf_counter() - t_start) / 60:.1f} min")
    for name in prefs:
        s = stats[name]
        print(f"  {name:<9s} {s['solves']} solves ({s['vi_time'] / 60:.1f} min VI), "
              f"rollout success {s['ok']}/{s['rollouts']} ({100 * s['ok'] / max(s['rollouts'], 1):.0f}%), "
              f"{sum(len(a) for a in cols[name]['state']):,} rows -> {out_path[name]}")


if __name__ == "__main__":
    main()
