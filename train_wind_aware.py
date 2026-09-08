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
    ap.add_argument("--obs", choices=["wind", "plain", "flat", "stencil"], default="wind",
                    help="wind: Dict obs with CNN maps; flat: the same maps flattened into one vector for an MLP; "
                         "plain: 6-vector obs with an MLP (wind-blind control); "
                         "stencil: 6-vector + wind on a small stencil around the ship (MLP)")
    ap.add_argument("--stencil-n", type=int, default=3)
    ap.add_argument("--stencil-spacing", type=float, default=1.0)
    ap.add_argument("--fixed-wind", default=None,
                    help="train and validate on ONE field: 'legacy' (WF.pkl) or an integer generator seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", default=None, help="path to a model .zip to continue training")
    ap.add_argument("--demo-data", default=None,
                    help="dp_teacher.npz: pre-fill the replay buffer with the DP rollout transitions")
    ap.add_argument("--bc-weight", type=float, default=0.0,
                    help="TD3+BC: weight of the behaviour-cloning MSE on demonstration samples in the actor "
                         "loss (0 = plain TD3). Requires --demo-data.")
    ap.add_argument("--bc-alpha", type=float, default=2.5,
                    help="TD3+BC: the Q term is scaled by alpha / mean|Q| (Fujimoto & Gu 2021)")
    ap.add_argument("--critic-warmup", type=int, default=0,
                    help="gradient steps during which the actor is frozen (critic learns first); "
                         "use together with --resume from a behaviour-cloned model")
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


def prefill_replay_buffer(model, data_path, obs_cfg, env_kwargs, n_envs):
    """Insert the DP rollout transitions of dp_teacher.npz into the model's replay buffer.
    Observations are rebuilt with the same wrapper; actions are stored scaled to [-1, 1]
    as SB3 does; rewards are the ones the env returned during the DP rollouts."""
    from env import ShipEnv
    from wind import generate_wind_field
    from wind_obs import wrap_wind_obs
    with np.load(data_path) as f:  # materialise once: NpzFile re-decompresses on every access
        d = {k: f[k] for k in ("field_seed", "goal", "state", "next_state", "action", "reward",
                               "terminated", "source")}
    rows = np.where(d["source"] == 0)[0]
    u_max = float(model.action_space.high[0])
    obs_l, nobs_l, act_l, rew_l, done_l = [], [], [], [], []
    for seed in np.unique(d["field_seed"][rows]):
        r = rows[d["field_seed"][rows] == seed]
        base = ShipEnv(generate_wind_field(int(seed)), **env_kwargs)
        wrapper = wrap_wind_obs(base, obs_cfg)
        # walk down to the WindObsWrapper (may be under FlattenObservation)
        obs_fn = wrapper.observation
        for i in r:
            base.goal = d["goal"][i].astype(np.float64)
            base.state = d["state"][i].astype(np.float64)
            obs_l.append(obs_fn(None))
            base.state = d["next_state"][i].astype(np.float64)
            nobs_l.append(obs_fn(None))
            act_l.append(d["action"][i] / u_max)
            rew_l.append(d["reward"][i])
            done_l.append(d["terminated"][i])
    n = (len(obs_l) // n_envs) * n_envs
    keys = obs_l[0].keys() if isinstance(obs_l[0], dict) else None
    if hasattr(model, "set_demonstrations"):
        if keys:
            demo_obs = {k: np.stack([o[k] for o in obs_l]) for k in keys}
        else:
            demo_obs = np.stack(obs_l)
        model.set_demonstrations(demo_obs, np.stack(act_l).astype(np.float32))
    for s in range(0, n, n_envs):
        if keys:
            o = {k: np.stack([obs_l[j][k] for j in range(s, s + n_envs)]) for k in keys}
            no = {k: np.stack([nobs_l[j][k] for j in range(s, s + n_envs)]) for k in keys}
        else:
            o = np.stack(obs_l[s:s + n_envs])
            no = np.stack(nobs_l[s:s + n_envs])
        model.replay_buffer.add(o, no, np.stack(act_l[s:s + n_envs]).astype(np.float32),
                                np.array(rew_l[s:s + n_envs], dtype=np.float32),
                                np.array(done_l[s:s + n_envs]), [{} for _ in range(n_envs)])
    return n


def _make_td3bc_class():
    import torch as th
    import torch.nn.functional as F
    from stable_baselines3 import TD3
    from stable_baselines3.common.utils import polyak_update

    class TD3BC(TD3):
        """
        TD3 whose actor loss adds a behaviour-cloning MSE on demonstration samples:
            L = -alpha / mean|Q| * Q(s, pi(s))  +  bc_weight * ||pi(s_demo) - a_demo||^2
        (Fujimoto & Gu, "A Minimalist Approach to Offline RL", 2021), used here for online
        fine-tuning of a cloned policy so the policy gradient cannot erase the expert before
        the critic is accurate. Demonstrations are set with `set_demonstrations`.
        """

        bc_weight = 1.0
        bc_alpha = 2.5

        def set_demonstrations(self, demo_obs, demo_actions):
            dev = self.device
            if isinstance(demo_obs, dict):
                self._demo_obs = {k: th.as_tensor(v, device=dev) for k, v in demo_obs.items()}
            else:
                self._demo_obs = th.as_tensor(demo_obs, device=dev)
            self._demo_act = th.as_tensor(demo_actions, device=dev)
            self._n_demo = len(self._demo_act)

        def _demo_batch(self, batch_size):
            i = th.randint(0, self._n_demo, (batch_size,), device=self.device)
            if isinstance(self._demo_obs, dict):
                return {k: v[i] for k, v in self._demo_obs.items()}, self._demo_act[i]
            return self._demo_obs[i], self._demo_act[i]

        def train(self, gradient_steps, batch_size=100):
            self.policy.set_training_mode(True)
            self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
            actor_losses, critic_losses, bc_losses = [], [], []
            for _ in range(gradient_steps):
                self._n_updates += 1
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                with th.no_grad():
                    noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                current_q_values = self.critic(replay_data.observations, replay_data.actions)
                critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
                critic_losses.append(critic_loss.item())
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                if self._n_updates % self.policy_delay == 0:
                    q = self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations))
                    lmbda = self.bc_alpha / q.abs().mean().detach().clamp_min(1e-6)
                    actor_loss = -lmbda * q.mean()
                    if self.bc_weight > 0 and getattr(self, "_n_demo", 0) > 0:
                        d_obs, d_act = self._demo_batch(batch_size)
                        bc = F.mse_loss(self.actor(d_obs), d_act)
                        actor_loss = actor_loss + self.bc_weight * bc
                        bc_losses.append(bc.item())
                    actor_losses.append(actor_loss.item())
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                    polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                    polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            if actor_losses:
                self.logger.record("train/actor_loss", np.mean(actor_losses))
            if bc_losses:
                self.logger.record("train/bc_loss", np.mean(bc_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))

        def _excluded_save_params(self):
            return super()._excluded_save_params() + ["_demo_obs", "_demo_act", "_n_demo"]

    return TD3BC


TD3BC = _make_td3bc_class()


def _make_actor_freeze_callback():
    from stable_baselines3.common.callbacks import BaseCallback

    class ActorFreezeCallback(BaseCallback):
        """Zero the actor learning rate for the first `n_updates` gradient steps."""

        def __init__(self, n_updates):
            super().__init__()
            self.n_updates = n_updates
            self._lr = None
            self.released = False

        def _on_training_start(self):
            opt = self.model.actor.optimizer
            self._lr = [g["lr"] for g in opt.param_groups]
            for g in opt.param_groups:
                g["lr"] = 0.0
            print(f"[warmup] actor frozen for the first {self.n_updates} gradient steps")

        def _on_step(self):
            if not self.released and self.model._n_updates >= self.n_updates:
                for g, lr in zip(self.model.actor.optimizer.param_groups, self._lr):
                    g["lr"] = lr
                self.released = True
                print(f"[warmup] actor released at {self.num_timesteps} env steps "
                      f"({self.model._n_updates} gradient steps)")
            return True

    return ActorFreezeCallback


ActorFreezeCallback = _make_actor_freeze_callback()


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
    if args.obs == "stencil":
        obs_cfg = dict(stencil=args.stencil_n, spacing=args.stencil_spacing)
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

    mlp = plain or args.obs in ("flat", "stencil")
    if mlp:
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    else:
        policy_kwargs = dict(
            features_extractor_class=WindCNNExtractor,
            features_extractor_kwargs=dict(map_features=64, vec_features=64),
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
            share_features_extractor=False,
        )
    common = dict(
        policy="MlpPolicy" if mlp else "MultiInputPolicy", env=train_env, learning_rate=args.lr, gamma=args.gamma,
        buffer_size=args.buffer_size, batch_size=args.batch_size, learning_starts=args.learning_starts,
        train_freq=1, gradient_steps=args.gradient_steps, policy_kwargs=policy_kwargs,
        seed=args.seed, device=args.device, verbose=0,
    )
    if args.resume:
        Algo = TD3 if args.algo == "td3" else SAC
        if args.bc_weight > 0:
            if not args.demo_data:
                raise SystemExit("--bc-weight requires --demo-data")
            Algo = TD3BC
        # a behaviour-cloned model was saved with placeholder RL hyper-parameters (tiny buffer,
        # no exploration noise): override them with this run's settings on load
        overrides = dict(learning_rate=args.lr, buffer_size=args.buffer_size, batch_size=args.batch_size,
                         learning_starts=args.learning_starts, gradient_steps=args.gradient_steps,
                         gamma=args.gamma, train_freq=1)
        model = Algo.load(args.resume, env=train_env, device=args.device, custom_objects=overrides)
        if args.algo == "td3":
            n_act = train_env.action_space.shape[-1]
            sigma = args.action_noise / float(train_env.action_space.high[0])
            model.action_noise = NormalActionNoise(mean=np.zeros(n_act), sigma=sigma * np.ones(n_act))
        if args.bc_weight > 0:
            model.bc_weight = args.bc_weight
            model.bc_alpha = args.bc_alpha
        print(f"resumed from {args.resume} with lr={args.lr}, buffer={args.buffer_size}, "
              f"noise={args.action_noise}, algo={type(model).__name__}, bc_weight={args.bc_weight}")
    elif args.algo == "td3":
        n_act = train_env.action_space.shape[-1]
        u_max = float(train_env.action_space.high[0])
        sigma = args.action_noise / u_max  # SB3 adds the noise to the scaled action in [-1, 1]
        noise = NormalActionNoise(mean=np.zeros(n_act), sigma=sigma * np.ones(n_act))
        model = TD3(action_noise=noise, **common)
    else:
        model = SAC(ent_coef="auto", **common)

    if args.demo_data:
        n_demo = prefill_replay_buffer(model, args.demo_data, obs_cfg, env_kwargs, args.n_envs)
        print(f"replay buffer pre-filled with {n_demo:,} DP demonstration transitions")

    model.set_logger(configure(log_dir, ["stdout", "csv"]))
    print(f"[{tag}] {args.algo.upper()} on {'ONE field (' + str(args.fixed_wind) + ')' if args.fixed_wind else str(args.n_train_fields) + ' train fields'}, "
          f"{args.n_envs} envs, device={model.device}, obs={'plain' if plain else obs_cfg}, "
          f"obs_dim={train_env.observation_space}, env={train_kwargs}")
    print(model.policy)

    callbacks = []
    if args.critic_warmup > 0:
        callbacks.append(ActorFreezeCallback(args.critic_warmup))
    callbacks += [
        EvalCallback(eval_env, best_model_save_path=None, log_path=log_dir,
                     eval_freq=max(args.eval_every // args.n_envs, 1),
                     n_eval_episodes=args.eval_episodes, deterministic=True, verbose=1),
        CheckpointCallback(save_freq=max(100_000 // args.n_envs, 1), save_path=log_dir,
                           name_prefix="ckpt", verbose=0),
    ]
    # EvalCallback saves best_model.zip into best_model_save_path; keep it next to the final model
    eval_cb = [c for c in callbacks if isinstance(c, EvalCallback)][0]
    eval_cb.best_model_save_path = os.path.join(MODEL_DIR, f"{tag}_best")
    os.makedirs(eval_cb.best_model_save_path, exist_ok=True)

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
