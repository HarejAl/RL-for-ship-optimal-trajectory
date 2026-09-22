"""
Score sailing policies against the DP optimum on fixed held-out cases.

    ref = dp_reference("uniform")              # solves DP once per case, cached on disk
    res = evaluate(policy, "uniform")          # policy(env: SailRLEnv) -> heading (radians)
    summary(res, ref)

Both the DP policy and the learned one are executed in the same `SailRLEnv` (dt = 0.1 h,
same polar, same tack/gybe penalties), so times compare on equal terms.
"""

import os
import time

import numpy as np

from sailing.rl_env import SailRLEnv, eval_cases, RL_PARAMS
from sailing.dp_sail import SailDP

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRIPT_DIR, "..", "output", "sailing", "rl")


def run_episode(env, policy, case, max_steps=600):
    wind, start, goal = case
    env.reset(options=dict(wind=wind, start=start, goal=goal))
    traj = [env.sail.state.copy()]
    info = env.sail._info()
    t0 = time.perf_counter()
    for _ in range(max_steps):
        h = policy(env)
        _, _, term, trunc, info = env.sail.step(np.array([h]))
        traj.append(env.sail.state.copy())
        if term or trunc:
            break
    return dict(t=info["t"], success=bool(info["success"]), tacks=info["tacks"], gybes=info["gybes"],
                traj=np.array(traj), wall_s=time.perf_counter() - t0)


def evaluate(policy, mode="uniform", n=40, env=None, keep_traj=False):
    env = env or SailRLEnv(None)
    out = []
    for case in eval_cases(mode, n):
        r = run_episode(env, policy, case)
        if not keep_traj:
            r.pop("traj")
        out.append(r)
    return out


def dp_reference(mode="uniform", n=40, nx=81, force=False):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"dp_ref_{mode}_{n}_{nx}.npz")
    if os.path.exists(path) and not force:
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    env = SailRLEnv(None)
    t, ok, tacks, gybes, solve = [], [], [], [], []
    for wind, start, goal in eval_cases(mode, n):
        dp = SailDP(wind, env.sail.polar, RL_PARAMS, goal, nx=nx, ny=nx)
        s = dp.solve()
        pol = dp.policy()
        r = run_episode(env, lambda e: pol(e.sail), (wind, start, goal))
        t.append(r["t"]); ok.append(r["success"]); tacks.append(r["tacks"]); gybes.append(r["gybes"])
        solve.append(s["solve_s"] + s["precompute_s"])
    ref = dict(t=np.array(t), success=np.array(ok), tacks=np.array(tacks), gybes=np.array(gybes),
               solve_s=np.array(solve))
    np.savez(path, **ref)
    return ref


def summary(res, ref):
    ok = np.array([r["success"] for r in res])
    t = np.array([r["t"] for r in res])
    both = ok & ref["success"]
    gap = (t[both] - ref["t"][both]) / ref["t"][both] * 100
    return dict(success=float(ok.mean()), dp_success=float(ref["success"].mean()),
                median_gap_pct=float(np.median(gap)) if both.any() else float("nan"),
                mean_gap_pct=float(np.mean(gap)) if both.any() else float("nan"),
                tacks=float(np.mean([r["tacks"] for r in res])), dp_tacks=float(ref["tacks"].mean()),
                gybes=float(np.mean([r["gybes"] for r in res])), dp_gybes=float(ref["gybes"].mean()),
                ms_per_episode=1000 * float(np.mean([r["wall_s"] for r in res])))


def load_agent(path, map_res=None, device="cpu"):
    """Load a model saved by train_rl.py (algorithm read from the file name) -> (policy, env)."""
    from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
    name = os.path.basename(path).lower()
    algo = (DQN if "dqn" in name else A2C if "a2c" in name else SAC if "sac" in name else
            TD3 if "td3" in name else PPO)
    model = algo.load(path, device=device)
    n_actions = getattr(model.action_space, "n", None)
    env = SailRLEnv(None, map_res=map_res, n_actions=int(n_actions) if n_actions else None)
    return sb3_policy(model), env


def sb3_policy(model):
    """Wrap an SB3 model trained on SailRLEnv observations as a heading policy."""
    def pol(env):
        a, _ = model.predict(env.observe(), deterministic=True)
        return env.heading_for(env.action_to_a(a))
    return pol


def goal_policy(env):
    """Baseline: always steer straight at the goal."""
    return env.heading_for(0.0)
