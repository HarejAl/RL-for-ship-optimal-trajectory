"""
Wind-aware observations and the CNN feature extractor for stable-baselines3.

`WindObsWrapper` turns the 6-vector observation of `ShipEnv` into a Dict:

    vec    : (6,)          [(xg-x)/10, (yg-y)/10, vx/6, vy/6, x/10, y/10]
    local  : (3, K, K)     ego-centric crop of half-width `local_size` around the ship:
                           wx/10, wy/10, inside-domain mask (axis-aligned, absolute frame)
    global : (4, G, G)     whole field downsampled to GxG: wx/10, wy/10, gaussian blob at
                           the ship, gaussian blob at the goal   (optional)

The local crop gives the policy the wind it is about to sail through; the coarse
global map lets it route around regions it cannot see locally. All channels are
scaled to O(1).

Speeds (ship velocity and wind) are divided by the ship's own calm-water speed V*
(`ShipParams.scales()`), expressed relative to the reference ship the policies were
trained on: the "6" and "10" above hold for the default `ShipParams()` and grow with V*.
For the default ship the observation is unchanged; for a faster or slower ship with
the same windage ratio kappa it is exactly the observation of the reference ship in
proportionally rescaled wind, so a trained policy transfers without retraining.

`WindFieldPool` holds generated fields with a fixed seed namespace, so training
fields (seeds >= TRAIN_SEED_BASE) never overlap the held-out benchmark fields used
by `benchmark_dp.py` (seeds < 1e6).

`WindCNNExtractor` is a `BaseFeaturesExtractor`: one small CNN per map input and an
MLP for the vector, concatenated.
"""

import dataclasses
import json
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from wind import generate_wind_field
from dynamics import ShipParams

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_SEED_BASE = 1_000_000   # training fields: seeds TRAIN_SEED_BASE + i
EVAL_SEED_BASE = 2_000_000    # validation fields used during training
POS_SCALE = 10.0
VEL_SCALE = 6.0               # for the reference ship; see speed_scales()
WIND_SCALE = 10.0
REFERENCE_SPEED = ShipParams().scales().speed   # V* of the ship every model was trained on
KAPPA_SCALE = 0.5             # windage ratio of the reference ship -> 1.0 in the observation


def speed_scales(params):
    """(velocity scale, wind scale) for a ship: the reference constants times V* / V*_ref."""
    k = params.scales().speed / REFERENCE_SPEED
    return VEL_SCALE * k, WIND_SCALE * k


class WindFieldPool:
    """Fixed set of generated wind fields; `sampler(rng)` picks one uniformly."""

    def __init__(self, n_fields, seed_base=TRAIN_SEED_BASE, **gen_kwargs):
        self.seeds = [seed_base + i for i in range(n_fields)]
        self.fields = [generate_wind_field(s, **gen_kwargs) for s in self.seeds]

    def sampler(self, rng):
        return self.fields[int(rng.integers(len(self.fields)))]

    def __len__(self):
        return len(self.fields)


class WindObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, local_size=2.0, local_res=16, global_res=16, use_global=True,
                 blob_sigma=0.6, add_kappa=False):
        super().__init__(env)
        self.local_size = float(local_size)
        self.local_res = int(local_res)
        self.global_res = int(global_res)
        self.use_global = bool(use_global)
        self.blob_sigma = float(blob_sigma)
        self.add_kappa = bool(add_kappa)   # append the windage ratio kappa / KAPPA_SCALE to `vec`

        off = np.linspace(-self.local_size, self.local_size, self.local_res)
        self._ox, self._oy = np.meshgrid(off, off, indexing="ij")
        self._global_cache = (None, None)

        d = {
            "vec": spaces.Box(-np.inf, np.inf, shape=(7 if self.add_kappa else 6,), dtype=np.float32),
            "local": spaces.Box(-np.inf, np.inf, shape=(3, self.local_res, self.local_res), dtype=np.float32),
        }
        if self.use_global:
            d["global"] = spaces.Box(-np.inf, np.inf, shape=(4, self.global_res, self.global_res),
                                     dtype=np.float32)
        self.observation_space = spaces.Dict(d)

    # ---------------------------------------------------------------- config
    def config(self):
        return dict(local_size=self.local_size, local_res=self.local_res,
                    global_res=self.global_res, use_global=self.use_global, blob_sigma=self.blob_sigma,
                    add_kappa=self.add_kappa)

    @staticmethod
    def save_config(path, cfg):
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)

    @staticmethod
    def load_config(path):
        with open(path) as f:
            return json.load(f)

    # ----------------------------------------------------------------- maps
    def _global_map(self, wind, wind_scale):
        """Downsampled wind components on a GxG grid over the field extent (cached per field
        and wind scale)."""
        key = self._global_cache[0]
        if key is not None and key[0] is wind and key[1] == wind_scale:
            return self._global_cache[1]
        xmin, xmax, ymin, ymax = wind.extent
        gx = np.linspace(xmin, xmax, self.global_res)
        gy = np.linspace(ymin, ymax, self.global_res)
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        wx, wy = wind(GX, GY)
        base = np.stack((wx / wind_scale, wy / wind_scale)).astype(np.float32)
        self._global_cache = ((wind, wind_scale), (base, GX, GY))
        return self._global_cache[1]

    def observation(self, obs):
        base = self.env.unwrapped
        wind = base.wind
        x, y, vx, vy = base.state
        gx, gy = base.goal
        xmin, xmax, ymin, ymax = wind.extent
        vel_scale, wind_scale = speed_scales(base.p)

        vec = np.array([(gx - x) / POS_SCALE, (gy - y) / POS_SCALE,
                        vx / vel_scale, vy / vel_scale, x / POS_SCALE, y / POS_SCALE], dtype=np.float32)
        if self.add_kappa:
            vec = np.append(vec, np.float32(base.p.scales().windage / KAPPA_SCALE))

        px = x + self._ox
        py = y + self._oy
        wx, wy = wind(px, py)
        inside = ((px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)).astype(np.float32)
        local = np.stack((wx / wind_scale, wy / wind_scale, inside)).astype(np.float32)

        out = {"vec": vec, "local": local}
        if self.use_global:
            wmap, GX, GY = self._global_map(wind, wind_scale)
            s2 = 2.0 * self.blob_sigma ** 2
            ship = np.exp(-((GX - x) ** 2 + (GY - y) ** 2) / s2).astype(np.float32)
            goal = np.exp(-((GX - gx) ** 2 + (GY - gy) ** 2) / s2).astype(np.float32)
            out["global"] = np.concatenate((wmap, ship[None], goal[None]), axis=0)
        return out


# ------------------------------------------------------------ SB3 extractor
def _make_extractor_class():
    import torch
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class WindCNNExtractor(BaseFeaturesExtractor):
        """CNN per map input + MLP for the vector input, concatenated."""

        def __init__(self, observation_space, map_features=64, vec_features=64):
            total = 0
            self.map_keys = [k for k in observation_space.spaces if k != "vec"]
            for k in self.map_keys:
                total += map_features
            total += vec_features
            super().__init__(observation_space, features_dim=total)

            self.cnns = nn.ModuleDict()
            for k in self.map_keys:
                c, h, w = observation_space[k].shape
                cnn = nn.Sequential(
                    nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Flatten(),
                )
                with torch.no_grad():
                    n_flat = cnn(torch.zeros(1, c, h, w)).shape[1]
                self.cnns[k] = nn.Sequential(cnn, nn.Linear(n_flat, map_features), nn.ReLU())
            n_vec = observation_space["vec"].shape[0]
            self.vec_mlp = nn.Sequential(nn.Linear(n_vec, vec_features), nn.ReLU())

        def forward(self, observations):
            feats = [self.cnns[k](observations[k]) for k in self.map_keys]
            feats.append(self.vec_mlp(observations["vec"]))
            return torch.cat(feats, dim=1)

    return WindCNNExtractor


WindCNNExtractor = _make_extractor_class()


class WindStencilWrapper(gym.ObservationWrapper):
    """
    Minimal wind-aware observation for an MLP: the 6-vector plus the wind (scaled) sampled
    on an n x n stencil of spacing `spacing` centred on the ship (absolute frame).
    obs dim = 6 + 2 * n * n.
    """

    def __init__(self, env, n=3, spacing=1.0):
        super().__init__(env)
        self.n = int(n)
        self.spacing = float(spacing)
        off = (np.arange(self.n) - (self.n - 1) / 2) * self.spacing
        self._ox, self._oy = np.meshgrid(off, off, indexing="ij")
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6 + 2 * self.n * self.n,), dtype=np.float32)

    def observation(self, obs):
        base = self.env.unwrapped
        x, y, vx, vy = base.state
        gx, gy = base.goal
        vel_scale, wind_scale = speed_scales(base.p)
        vec = np.array([(gx - x) / POS_SCALE, (gy - y) / POS_SCALE,
                        vx / vel_scale, vy / vel_scale, x / POS_SCALE, y / POS_SCALE], dtype=np.float32)
        wx, wy = base.wind(x + self._ox, y + self._oy)
        return np.concatenate((vec, (wx / wind_scale).ravel(), (wy / wind_scale).ravel())).astype(np.float32)


class ReferenceThrustWrapper(gym.ActionWrapper):
    """
    Exposes the reference ship's action space (|u_i| <= u_max of `ShipParams()`) on a ship
    with a different thrust bound, scaling the action by u_max / u_max_ref. Policies are
    trained on the reference ship, so this is what lets them command another ship: together
    with the V*-normalised observation, a faster ship with the same windage ratio sees and
    acts exactly as the reference ship in proportionally weaker wind.
    """

    def __init__(self, env):
        super().__init__(env)
        ref = ShipParams().u_max
        self.factor = env.unwrapped.p.u_max / ref
        self.action_space = spaces.Box(-ref, ref, shape=env.action_space.shape, dtype=env.action_space.dtype)

    def action(self, action):
        return np.asarray(action, dtype=np.float64) * self.factor


class _PerceivedView:
    """What the policy is told: the real env, with the wind and ship swapped for perceived ones."""

    def __init__(self, base, wind, params):
        self._base, self.wind, self.p = base, wind, params

    def __getattr__(self, name):
        return getattr(self._base, name)


class PerceivedWindWrapper(gym.Wrapper):
    """
    Deployment-time windage mapping: a policy trained on a ship of windage ratio kappa_ref is
    shown the wind scaled by sqrt(kappa / kappa_ref), so a ship more prone to being pushed sees
    a stronger map, while the dynamics keep the true wind and the true ship. The scaling matches
    the wind force on a ship at rest; it is approximate once the ship moves (c_a also drags on
    the ship's own motion through the air). Put it between ShipEnv and WindObsWrapper.
    """

    def __init__(self, env, kappa_ref=0.5):
        from wind import WindField
        super().__init__(env)
        base = env.unwrapped
        k = base.p.scales().windage
        self.factor = float(np.sqrt(k / kappa_ref))
        self.ref_params = dataclasses.replace(base.p, cd_air=kappa_ref * base.p.cd_water)
        self._WindField = WindField
        self._cache = (None, None)

    @property
    def unwrapped(self):
        base = self.env.unwrapped
        if self._cache[0] is not base.wind:
            w = base.wind
            self._cache = (w, self._WindField(w.x, w.y, self.factor * w.wx, self.factor * w.wy, meta=w.meta))
        return _PerceivedView(base, self._cache[1], self.ref_params)


def wrap_wind_obs(base_env, obs_cfg):
    """Apply the observation wrapper described by obs_cfg to a ShipEnv:
    {'stencil': n, 'spacing': s}  -> WindStencilWrapper (flat, for MlpPolicy)
    otherwise WindObsWrapper(**cfg), plus FlattenObservation if cfg['flatten'].
    A ship whose thrust bound differs from the reference also gets ReferenceThrustWrapper."""
    cfg = dict(obs_cfg or {})
    if base_env.unwrapped.p.u_max != ShipParams().u_max:
        base_env = ReferenceThrustWrapper(base_env)
    if cfg.get("stencil"):
        return WindStencilWrapper(base_env, n=cfg["stencil"], spacing=cfg.get("spacing", 1.0))
    flatten = cfg.pop("flatten", False)
    env = WindObsWrapper(base_env, **cfg)
    if flatten:
        env = gym.wrappers.FlattenObservation(env)
    return env


def make_wind_env(pool=None, wind=None, obs_cfg=None, env_kwargs=None, monitor=True):
    """ShipEnv + WindObsWrapper (+ FlattenObservation) (+ Monitor).
    Either a pool (sampled per reset) or a fixed wind. obs_cfg may contain 'flatten': True
    to get a flat Box observation for MlpPolicy instead of the Dict for the CNN extractor."""
    from env import ShipEnv
    env_kwargs = env_kwargs or {}
    if pool is not None:
        base = ShipEnv(wind_sampler=pool.sampler, **env_kwargs)
    else:
        base = ShipEnv(wind, **env_kwargs)
    env = wrap_wind_obs(base, obs_cfg)
    if monitor:
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env, info_keywords=("success", "J", "t", "goal_radius"))
    return env
