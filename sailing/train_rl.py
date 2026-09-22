"""
Pure RL (SAC, no teacher) for sailing routes, scored against DP during training.

    python sailing/train_rl.py --mode uniform --tag sac_uniform --steps 300000
    python sailing/train_rl.py --mode fields  --tag sac_fields  --steps 600000 --map-res 16
    python sailing/train_rl.py --mode uniform --tag ppo_uniform --algo ppo --n-actions 36 --steps 2000000
    python sailing/train_rl.py --mode regatta --tag ppo_regatta_blind --algo ppo --n-actions 36 --steps 3000000
    python sailing/train_rl.py --mode regatta --tag ppo_regatta_map --algo ppo --n-actions 36 --map-res 16 --steps 3000000
    python sailing/train_rl.py --mode regatta --tag a2c_time --algo a2c --n-actions 36 --shaping time --steps 2000000

Algorithms: ppo / a2c / dqn need --n-actions (discrete headings); sac / td3 steer continuously.
Shaping: dist (closer is better), time (hours-to-go through the polar), none (pure time reward).

uniform : constant wind of random direction/strength -> MLP on the 10-vector
fields  : random spatially varying fields -> CNN on the goal-aligned wind map + MLP on the vector

Every --eval-every steps the deterministic policy sails the fixed held-out cases; the log and
models/sail/<tag>_best.zip track success rate and median time gap to DP.
"""

import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, REPO_DIR)

import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from sailing.rl_env import make_rl_env, SailRLEnv
from sailing.rl_eval import evaluate, dp_reference, summary, sb3_policy, goal_policy
from wind_obs import WindCNNExtractor

MODEL_DIR = os.path.join(REPO_DIR, "models", "sail")
LOG_DIR = os.path.join(REPO_DIR, "output", "sailing", "rl")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("uniform", "fields", "regatta"), default="uniform")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--algo", choices=("ppo", "a2c", "dqn", "sac", "td3"), default="ppo")
    ap.add_argument("--n-actions", type=int, default=None, help="discrete headings (required for ppo here)")
    ap.add_argument("--map-res", type=int, default=None)
    ap.add_argument("--n-fields", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--shaping", choices=("dist", "time", "none"), default="dist")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


class DPEvalCallback(BaseCallback):
    def __init__(self, args, ref, log_path, best_path):
        super().__init__()
        self.args, self.ref, self.log_path, self.best_path = args, ref, log_path, best_path
        self.best = (-1.0, -np.inf)
        self.next_eval = args.eval_every

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.args.eval_every
        env = SailRLEnv(None, map_res=self.args.map_res, n_actions=self.args.n_actions)
        res = evaluate(sb3_policy(self.model), self.args.mode, self.args.eval_n, env=env)
        s = summary(res, self.ref)
        ep = list(getattr(self.model, "ep_info_buffer", []) or [])
        s.update(steps=self.num_timesteps, wall_min=(time.time() - self.t_start) / 60,
                 train_reward=float(np.mean([e["r"] for e in ep])) if ep else float("nan"),
                 train_ep_len=float(np.mean([e["l"] for e in ep])) if ep else float("nan"))
        with open(self.log_path, "a") as f:
            f.write(json.dumps(s) + "\n")
        print(f"[{self.num_timesteps:>7}] success {s['success']:.0%} (DP {s['dp_success']:.0%})  "
              f"median gap {s['median_gap_pct']:+.1f}%  tacks {s['tacks']:.1f} (DP {s['dp_tacks']:.1f})  "
              f"gybes {s['gybes']:.1f} (DP {s['dp_gybes']:.1f})  train R {s['train_reward']:+.1f}  "
              f"{s['wall_min']:.0f} min", flush=True)
        gap = s["median_gap_pct"] if np.isfinite(s["median_gap_pct"]) else 1e9
        key = (s["success"], -gap)
        if key >= self.best:
            self.best = key
            self.model.save(self.best_path)
        return True

    def _on_training_start(self):
        self.t_start = time.time()


def main():
    args = parse_args()
    tag = args.tag or f"sac_{args.mode}"
    torch.set_num_threads(2)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("DP reference on held-out cases ...", flush=True)
    ref = dp_reference(args.mode, args.eval_n)
    base = summary(evaluate(goal_policy, args.mode, args.eval_n), ref)
    print(f"DP: success {ref['success'].mean():.0%}, mean {ref['t'].mean():.1f} h, tacks {ref['tacks'].mean():.1f}, "
          f"solve {ref['solve_s'].mean():.2f} s | steer-at-goal: success {base['success']:.0%}, "
          f"gap {base['median_gap_pct']:+.1f}%", flush=True)

    env_kw = dict(mode=args.mode, n_fields=args.n_fields, map_res=args.map_res, gamma=args.gamma,
                  shaping=args.shaping, n_actions=args.n_actions)

    def make(i):
        return lambda: make_rl_env(seed=args.seed * 100 + i, **env_kw)

    vec = SubprocVecEnv([make(i) for i in range(args.n_envs)]) if args.n_envs > 1 else DummyVecEnv([make(0)])
    if args.map_res:
        policy, pk = "MultiInputPolicy", dict(features_extractor_class=WindCNNExtractor, net_arch=[256, 256])
    else:
        policy, pk = "MlpPolicy", dict(net_arch=[256, 256])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # small MLPs run faster on the CPU; only the CNN runs (a lot) faster on the GPU
    dev = device if args.map_res else "cpu"
    common = dict(learning_rate=args.lr, gamma=args.gamma, policy_kwargs=pk, seed=args.seed,
                  verbose=0, device=dev)
    if args.algo == "ppo":
        model = PPO(policy, vec, n_steps=512, batch_size=512, n_epochs=10, gae_lambda=0.95,
                    ent_coef=0.01, **common)
    elif args.algo == "a2c":
        model = A2C(policy, vec, n_steps=32, gae_lambda=0.95, ent_coef=0.01, **common)
    elif args.algo == "dqn":
        model = DQN(policy, vec, buffer_size=300_000, batch_size=256, learning_starts=10_000,
                    train_freq=4, gradient_steps=1, target_update_interval=2_000,
                    exploration_fraction=0.3, exploration_final_eps=0.05, **common)
    elif args.algo == "td3":
        from stable_baselines3.common.noise import NormalActionNoise
        model = TD3(policy, vec, buffer_size=500_000, batch_size=256, learning_starts=5_000,
                    train_freq=1, gradient_steps=2,
                    action_noise=NormalActionNoise(np.zeros(1), 0.15 * np.ones(1)), **common)
    else:
        model = SAC(policy, vec, buffer_size=500_000, batch_size=256, learning_starts=5_000,
                    train_freq=1, gradient_steps=2, **common)

    log_path = os.path.join(LOG_DIR, f"{tag}_log.jsonl")
    open(log_path, "w").close()
    cb = DPEvalCallback(args, ref, log_path, os.path.join(MODEL_DIR, f"{tag}_best"))
    with open(os.path.join(MODEL_DIR, f"{tag}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    model.learn(total_timesteps=args.steps, callback=cb)
    model.save(os.path.join(MODEL_DIR, tag))
    print("done", flush=True)


if __name__ == "__main__":
    main()
