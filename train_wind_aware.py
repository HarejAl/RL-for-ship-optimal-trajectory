"""
Train a wind-aware policy (CNN over the wind map) on a pool of generated wind fields.

    python train_wind_aware.py --timesteps 1000000 --n-envs 8 --tag td3_v1
    python train_wind_aware.py --algo sac --timesteps 500000 --tag sac_v1

Outputs
    models/<tag>.zip           final model
    models/<tag>_best.zip      best model on the validation fields (EvalCallback)
    models/<tag>.json          observation-wrapper config (needed to rebuild the env)
    output/logs/<tag>/         SB3 CSV/stdout logs and evaluations.npz
"""

import argparse
import functools
import os

import numpy as np
import torch

from wind_obs import (WindFieldPool, WindCNNExtractor, make_wind_env,
                      TRAIN_SEED_BASE, EVAL_SEED_BASE)

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
LOG_DIR = os.path.join(SCRIPT_DIR, "output", "logs")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=["td3", "sac"], default="td3")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-train-fields", type=int, default=200)
    ap.add_argument("--n-eval-fields", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=25_000, help="env steps (per env) between evaluations")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--buffer-size", type=int, default=300_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--learning-starts", type=int, default=10_000)
    ap.add_argument("--gradient-steps", type=int, default=-1, help="-1: one per collected transition")
    ap.add_argument("--action-noise", type=float, default=1.5,
                    help="TD3 exploration noise std in action units (SB3 applies it in the scaled [-1,1] space, "
                         "so it is divided by u_max internally)")
    ap.add_argument("--goal-bonus", type=float, default=10.0)
    ap.add_argument("--oob-penalty", type=float, default=10.0)
    ap.add_argument("--target-w", type=float, default=1.0, help="distance shaping weight")
    ap.add_argument("--curriculum", type=float, nargs=4, default=None, metavar=("R0", "THRESH", "SHRINK", "WINDOW"),
                    help="adaptive goal-radius curriculum for the training envs, e.g. 1.0 0.7 0.8 50")
    ap.add_argument("--local-size", type=float, default=2.0)
    ap.add_argument("--local-res", type=int, default=16)
    ap.add_argument("--global-res", type=int, default=16)
    ap.add_argument("--no-global", action="store_true")
    ap.add_argument("--obs", choices=["wind", "plain", "flat"], default="wind",
                    help="wind: Dict obs with CNN maps; flat: the same maps flattened into one vector for an MLP; "
                         "plain: 6-vector obs with an MLP (wind-blind control)")
    ap.add_argument("--fixed-wind", default=None,
                    help="train and validate on ONE field: 'legacy' (WF.pkl) or an integer generator seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", default=None, help="path to a model .zip to continue training")
    return ap.parse_args()


def _env_factory(n_fields, seed_base, obs_cfg, env_kwargs, fixed_wind=None, plain=False):
    from stable_baselines3.common.monitor import Monitor
    if fixed_wind is not None:
        from wind import WindField, generate_wind_field
        wind = WindField.load_legacy() if fixed_wind == "legacy" else generate_wind_field(int(fixed_wind))
        pool = None
    else:
        wind = None
        pool = WindFieldPool(n_fields, seed_base=seed_base)
    if plain:
        from env import ShipEnv
        env = ShipEnv(wind, wind_sampler=None if pool is None else pool.sampler, **env_kwargs)
        return Monitor(env, info_keywords=("success", "J", "t", "goal_radius"))
    return make_wind_env(pool=pool, wind=wind, obs_cfg=obs_cfg, env_kwargs=env_kwargs)


def main():
    args = parse_args()
    tag = args.tag or f"{args.algo}_s{args.seed}"
    os.makedirs(MODEL_DIR, exist_ok=True)
    log_dir = os.path.join(LOG_DIR, tag)
    os.makedirs(log_dir, exist_ok=True)

    from stable_baselines3 import TD3, SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.logger import configure

    obs_cfg = dict(local_size=args.local_size, local_res=args.local_res,
                   global_res=args.global_res, use_global=not args.no_global)
    if args.obs == "flat":
        obs_cfg["flatten"] = True
    from wind_obs import WindObsWrapper
    WindObsWrapper.save_config(os.path.join(MODEL_DIR, f"{tag}.json"), obs_cfg)

    env_kwargs = dict(goal_bonus=args.goal_bonus, oob_penalty=args.oob_penalty, target_w=args.target_w)
    train_kwargs = dict(env_kwargs)
    if args.curriculum:
        r0, thr, shrink, window = args.curriculum
        train_kwargs["curriculum"] = (r0, thr, shrink, int(window))
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    plain = args.obs == "plain"
    train_env = make_vec_env(
        functools.partial(_env_factory, args.n_train_fields, TRAIN_SEED_BASE, obs_cfg, train_kwargs,
                          args.fixed_wind, plain),
        n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls,
    )
    eval_env = make_vec_env(
        functools.partial(_env_factory, args.n_eval_fields, EVAL_SEED_BASE, obs_cfg, env_kwargs,
                          args.fixed_wind, plain),
        n_envs=1, seed=args.seed + 12345, vec_env_cls=DummyVecEnv,
    )

    if plain or args.obs == "flat":
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    else:
        policy_kwargs = dict(
            features_extractor_class=WindCNNExtractor,
            features_extractor_kwargs=dict(map_features=64, vec_features=64),
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
            share_features_extractor=False,
        )
    common = dict(
        policy="MlpPolicy" if (plain or args.obs == "flat") else "MultiInputPolicy", env=train_env, learning_rate=args.lr, gamma=args.gamma,
        buffer_size=args.buffer_size, batch_size=args.batch_size, learning_starts=args.learning_starts,
        train_freq=1, gradient_steps=args.gradient_steps, policy_kwargs=policy_kwargs,
        seed=args.seed, device=args.device, verbose=0,
    )
    if args.resume:
        Algo = TD3 if args.algo == "td3" else SAC
        model = Algo.load(args.resume, env=train_env, device=args.device)
        print(f"resumed from {args.resume}")
    elif args.algo == "td3":
        n_act = train_env.action_space.shape[-1]
        u_max = float(train_env.action_space.high[0])
        sigma = args.action_noise / u_max  # SB3 adds the noise to the scaled action in [-1, 1]
        noise = NormalActionNoise(mean=np.zeros(n_act), sigma=sigma * np.ones(n_act))
        model = TD3(action_noise=noise, **common)
    else:
        model = SAC(ent_coef="auto", **common)

    model.set_logger(configure(log_dir, ["stdout", "csv"]))
    print(f"[{tag}] {args.algo.upper()} on {'ONE field (' + str(args.fixed_wind) + ')' if args.fixed_wind else str(args.n_train_fields) + ' train fields'}, "
          f"{args.n_envs} envs, device={model.device}, obs={'plain' if plain else obs_cfg}, "
          f"obs_dim={train_env.observation_space}, env={train_kwargs}")
    print(model.policy)

    callbacks = [
        EvalCallback(eval_env, best_model_save_path=None, log_path=log_dir,
                     eval_freq=max(args.eval_every // args.n_envs, 1),
                     n_eval_episodes=args.eval_episodes, deterministic=True, verbose=1),
        CheckpointCallback(save_freq=max(100_000 // args.n_envs, 1), save_path=log_dir,
                           name_prefix="ckpt", verbose=0),
    ]
    # EvalCallback saves best_model.zip into best_model_save_path; keep it next to the final model
    callbacks[0].best_model_save_path = os.path.join(MODEL_DIR, f"{tag}_best")
    os.makedirs(callbacks[0].best_model_save_path, exist_ok=True)

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False,
                reset_num_timesteps=args.resume is None)
    final = os.path.join(MODEL_DIR, f"{tag}.zip")
    model.save(final)
    print(f"saved {final}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    torch.set_num_threads(2)  # GPU does the training; keep CPU threads low to coexist with other jobs
    main()
