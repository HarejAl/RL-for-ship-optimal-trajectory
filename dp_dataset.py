"""
Build a teacher dataset from the DP baseline for imitation learning.

For every training wind field (seeds TRAIN_SEED_BASE + i) and `--goals-per-field`
random goals, value iteration is solved once and then used to label
  * states along DP rollouts from random starts (with the real env transitions, so
    they can also seed a replay buffer), and
  * uniformly random states (position in the domain, small random velocity),
with the greedy DP action and the cost-to-go V(s).

    python dp_dataset.py --n-fields 150 --goals-per-field 2 --rollouts 20 --random-states 1500

Output: data/dp_teacher.npz with arrays
    field_seed (N,)  goal (N,2)  state (N,4)  action (N,2)  value (N,)  source (N,) 0=DP rollout 1=random 2=DAgger rollout
    and, for rollout samples only (source==0), the transition
    next_state (N,4)  reward (N,)  terminated (N,)   (NaN / 0 for random samples)
The file is rewritten after every field so a partial run is usable.
"""

import argparse
import os
import time

import numpy as np

from dynamics import ShipParams
from env import ShipEnv
from wind import generate_wind_field
from dp_baseline import ValueIterationPlanner
from wind_obs import TRAIN_SEED_BASE

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-fields", type=int, default=150)
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "dp_teacher.npz"))
    ap.add_argument("--rollout-policy", default=None,
                    help="DAgger: roll out this SB3 model (wind-aware) instead of the DP policy and label the "
                         "visited states with the DP action; the transition fields are then NOT expert transitions")
    ap.add_argument("--rollout-noise", type=float, default=0.0,
                    help="std (action units) of Gaussian noise added to the rolled-out policy (DAgger diversity)")
    return ap.parse_args()


def policy_rollout(model, env, start, goal, wind, rng, noise_std=0.0):
    """Roll out an SB3 policy (optionally with Gaussian action noise); same return format as planner.rollout."""
    base = env.unwrapped
    obs, info = env.reset(options=dict(start=start, goal=goal, wind=wind))
    traj, actions, rewards = [base.state.copy()], [], []
    terminated = truncated = False
    u_max = base.p.u_max
    for _ in range(base.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        if noise_std > 0:
            a = np.clip(a + rng.normal(0.0, noise_std, size=a.shape), -u_max, u_max)
        obs, r, terminated, truncated, info = env.step(a)
        traj.append(base.state.copy())
        actions.append(np.asarray(a, dtype=np.float64))
        rewards.append(r)
        if terminated or truncated:
            break
    return dict(J=info["J"], t=info["t"], success=info["success"], oob=info["oob"], steps=len(actions),
                terminated=bool(terminated), traj=np.array(traj), actions=np.array(actions),
                rewards=np.array(rewards))


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    params = ShipParams()
    rng = np.random.default_rng(args.seed)
    policy, obs_cfg = None, None
    if args.rollout_policy:
        from benchmark_dp import load_model
        policy, obs_cfg = load_model(args.rollout_policy)
        print(f"DAgger mode: rolling out {args.rollout_policy} (obs_cfg={obs_cfg}), labelling with DP")
    cols = {k: [] for k in ("field_seed", "goal", "state", "action", "value", "source",
                            "next_state", "reward", "terminated")}
    t_start = time.perf_counter()
    n_solves = 0
    solve_time = 0.0

    for i in range(args.first_field, args.first_field + args.n_fields):
        seed = TRAIN_SEED_BASE + i
        wind = generate_wind_field(seed)
        env = ShipEnv(wind, params=params, goal_bonus=args.goal_bonus, oob_penalty=args.oob_penalty)
        xmin, xmax, ymin, ymax = wind.extent
        rl_env = env
        if policy is not None and obs_cfg is not None:
            from wind_obs import wrap_wind_obs
            rl_env = wrap_wind_obs(env, obs_cfg)
        for g in range(args.goals_per_field):
            goal = rng.uniform(*env.spawn_box, size=2)
            planner = ValueIterationPlanner(wind, goal, params=params, nx=args.nx, ny=args.ny, nv=args.nv,
                                            n_act=args.n_act, exec_n_act=args.exec_n_act, device=args.device)
            stats = planner.solve(verbose=False)
            n_solves += 1
            solve_time += stats["time"]

            # --- rollouts from random starts: states, greedy actions, real transitions
            n_ok = 0
            for r in range(args.rollouts):
                start = rng.uniform(*env.spawn_box, size=2)
                while np.linalg.norm(start - goal) < env.min_start_goal_dist:
                    start = rng.uniform(*env.spawn_box, size=2)
                if policy is None:
                    res = planner.rollout(env, start)
                    labels = res["actions"]
                else:
                    res = policy_rollout(policy, rl_env, start, goal, wind, rng, args.rollout_noise)
                    labels, _ = planner.act_batch(res["traj"][:-1])
                n_ok += int(res["success"])
                T = res["steps"]
                st = res["traj"][:-1]
                cols["field_seed"].append(np.full(T, seed))
                cols["goal"].append(np.tile(goal, (T, 1)))
                cols["state"].append(st)
                cols["action"].append(labels)
                cols["value"].append(planner.value(st[:, 0], st[:, 1], st[:, 2], st[:, 3]))
                # source 0 = expert (DP) rollout transitions; 2 = DAgger rollouts of a learner policy
                # (labels are DP actions but the transitions follow the learner, so not demo transitions)
                cols["source"].append(np.full(T, 0 if policy is None else 2, dtype=np.int8))
                cols["next_state"].append(res["traj"][1:])
                cols["reward"].append(res["rewards"])
                term = np.zeros(T, dtype=bool)
                term[-1] = res["terminated"]
                cols["terminated"].append(term)

            # --- random states labelled with the greedy action
            n = args.random_states
            S = np.empty((n, 4))
            S[:, 0] = rng.uniform(xmin, xmax, n)
            S[:, 1] = rng.uniform(ymin, ymax, n)
            S[:, 2:] = np.clip(rng.normal(0.0, args.vel_std, (n, 2)), -planner.v_max, planner.v_max)
            A, Q = planner.act_batch(S)
            cols["field_seed"].append(np.full(n, seed))
            cols["goal"].append(np.tile(goal, (n, 1)))
            cols["state"].append(S)
            cols["action"].append(A)
            cols["value"].append(planner.value(S[:, 0], S[:, 1], S[:, 2], S[:, 3]))
            cols["source"].append(np.ones(n, dtype=np.int8))
            cols["next_state"].append(np.full((n, 4), np.nan))
            cols["reward"].append(np.zeros(n))
            cols["terminated"].append(np.zeros(n, dtype=bool))

            print(f"field {i:3d} goal {g}  solve {stats['time']:5.1f}s ({stats['iterations']} it)  "
                  f"rollouts ok {n_ok}/{args.rollouts}  samples so far {sum(len(a) for a in cols['state']):,}  "
                  f"elapsed {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)

        # rewrite the file after each field so partial runs are usable
        np.savez_compressed(args.out, **{k: np.concatenate(v) for k, v in cols.items()},
                            meta=np.array([n_solves, solve_time, args.nx, args.ny, args.nv]))

    print(f"done: {n_solves} solves, {solve_time / 60:.1f} min of VI, "
          f"{sum(len(a) for a in cols['state']):,} samples -> {args.out}")


if __name__ == "__main__":
    main()
