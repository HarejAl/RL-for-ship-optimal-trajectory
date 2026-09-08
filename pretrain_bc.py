"""
Behaviour cloning of the DP teacher into the wind-aware CNN actor.

Loads data/dp_teacher.npz (from dp_dataset.py), rebuilds the WindObsWrapper observation
of every labelled state, and regresses the SB3 TD3 actor onto the DP action. The result
is saved as a regular SB3 model (models/<tag>.zip + <tag>.json) that train_wind_aware.py
can resume from (RL fine-tuning) and benchmark_dp.py can evaluate.

    python pretrain_bc.py --tag bc_v1 --epochs 15
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from env import ShipEnv
from wind import generate_wind_field
from wind_obs import (WindObsWrapper, WindCNNExtractor, WindFieldPool, make_wind_env,
                      EVAL_SEED_BASE)

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(DATA_DIR, "dp_teacher.npz"))
    ap.add_argument("--tag", default="bc_v1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--local-size", type=float, default=2.0)
    ap.add_argument("--local-res", type=int, default=16)
    ap.add_argument("--global-res", type=int, default=16)
    ap.add_argument("--no-global", action="store_true")
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def build_observations(data, obs_cfg, idx):
    """Rebuild the wrapper observation for dataset rows `idx`; returns dict of arrays."""
    seeds = data["field_seed"][idx]
    states = data["state"][idx]
    goals = data["goal"][idx]
    out = None
    order = np.argsort(seeds, kind="stable")
    t0 = time.perf_counter()
    for seed in np.unique(seeds):
        rows = order[np.searchsorted(seeds[order], seed, "left"):np.searchsorted(seeds[order], seed, "right")]
        wind = generate_wind_field(int(seed))
        base = ShipEnv(wind)
        wrapper = WindObsWrapper(base, **obs_cfg)
        base.wind = wind
        for r in rows:
            base.state = states[r].astype(np.float64)
            base.goal = goals[r].astype(np.float64)
            o = wrapper.observation(None)
            if out is None:
                out = {k: np.empty((len(idx),) + v.shape, dtype=np.float32) for k, v in o.items()}
            for k, v in o.items():
                out[k][r] = v
    print(f"built {len(idx):,} observations in {time.perf_counter() - t0:.0f}s")
    return out


def evaluate(model, obs_cfg, n_episodes, seed):
    pool = WindFieldPool(20, seed_base=EVAL_SEED_BASE)
    env = make_wind_env(pool=pool, obs_cfg=obs_cfg, monitor=False)
    succ, J, T = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        for _ in range(env.unwrapped.max_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        succ.append(info["success"])
        J.append(info["J"])
        T.append(info["t"])
    succ = np.array(succ)
    return dict(success=succ.mean(), J_succ=float(np.mean(np.array(J)[succ])) if succ.any() else np.nan,
                t_succ=float(np.mean(np.array(T)[succ])) if succ.any() else np.nan)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(MODEL_DIR, exist_ok=True)
    obs_cfg = dict(local_size=args.local_size, local_res=args.local_res,
                   global_res=args.global_res, use_global=not args.no_global)

    data = np.load(args.data)
    N = len(data["state"])
    idx = np.arange(N)
    if args.max_samples and N > args.max_samples:
        idx = rng.choice(N, args.max_samples, replace=False)
    print(f"dataset {args.data}: {N:,} rows, using {len(idx):,}  "
          f"(rollout {int((data['source'][idx] == 0).sum()):,}, random {int((data['source'][idx] == 1).sum()):,})")

    obs = build_observations(data, obs_cfg, idx)
    from dynamics import ShipParams
    u_max = ShipParams().u_max
    target = (data["action"][idx] / u_max).astype(np.float32)  # actor outputs in [-1, 1]

    # train/val split by field so validation measures generalisation to unseen fields
    seeds = data["field_seed"][idx]
    uniq = np.unique(seeds)
    val_seeds = rng.choice(uniq, max(1, int(len(uniq) * args.val_frac)), replace=False)
    val_mask = np.isin(seeds, val_seeds)
    tr, va = np.where(~val_mask)[0], np.where(val_mask)[0]
    print(f"train {len(tr):,} samples / {len(uniq) - len(val_seeds)} fields, "
          f"val {len(va):,} samples / {len(val_seeds)} fields")

    # SB3 model with the same policy as train_wind_aware.py
    from stable_baselines3 import TD3
    env = make_wind_env(pool=WindFieldPool(1, seed_base=EVAL_SEED_BASE), obs_cfg=obs_cfg)
    policy_kwargs = dict(features_extractor_class=WindCNNExtractor,
                         features_extractor_kwargs=dict(map_features=64, vec_features=64),
                         net_arch=dict(pi=[256, 256], qf=[256, 256]), share_features_extractor=False)
    model = TD3("MultiInputPolicy", env, buffer_size=1000, policy_kwargs=policy_kwargs,
                device=args.device, seed=args.seed, verbose=0)
    actor = model.actor
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    dev = model.device

    def batch(ids):
        return ({k: torch.as_tensor(v[ids], device=dev) for k, v in obs.items()},
                torch.as_tensor(target[ids], device=dev))

    def val_loss():
        actor.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(va), 4096):
                o, y = batch(va[s:s + 4096])
                tot += F.mse_loss(actor(o), y, reduction="sum").item()
                n += len(y)
        actor.train()
        return tot / max(n, 1) / 2  # per-sample MSE averaged over the 2 action dims

    print(f"val MSE before training: {val_loss():.4f}")
    steps_per_epoch = int(np.ceil(len(tr) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(tr)
        t0, run = time.perf_counter(), 0.0
        for s in range(0, len(perm), args.batch_size):
            o, y = batch(perm[s:s + args.batch_size])
            loss = F.mse_loss(actor(o), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * len(y)
        print(f"epoch {ep:3d}  train MSE {run / len(perm):.4f}  val MSE {val_loss():.4f}  "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    model.actor_target.load_state_dict(model.actor.state_dict())
    out = os.path.join(MODEL_DIR, f"{args.tag}.zip")
    model.save(out)
    WindObsWrapper.save_config(os.path.join(MODEL_DIR, f"{args.tag}.json"), obs_cfg)
    print(f"saved {out}")

    res = evaluate(model, obs_cfg, args.eval_episodes, seed=args.seed + 777)
    print(f"held-out rollout (fixed goal radius): success {res['success'] * 100:.0f}%  "
          f"mean J on successes {res['J_succ']:.2f}  mean T {res['t_succ']:.2f}s")


if __name__ == "__main__":
    main()
