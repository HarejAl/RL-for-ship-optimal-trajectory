"""
Actor-critic network of the pure-RL agent (PLAN.md step B2), and a `.predict` adapter so a
trained checkpoint plugs into the CPU tools (`benchmark_dp.py`, `compare_policy.py`).

Observation = `WindObsWrapper`: vec (6,) or (7,) with kappa, local (3, K, K), global (4, G, G).
Actor and critic have separate encoders (same layout as `WindCNNExtractor`), so the value
regression cannot disturb the policy features. Actions are normalised to [-1, 1] per axis.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn


def _cnn(c, h, w, out):
    net = nn.Sequential(
        nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(),
    )
    with torch.no_grad():
        n = net(torch.zeros(1, c, h, w)).shape[1]
    return nn.Sequential(net, nn.Linear(n, out), nn.ReLU())


class Encoder(nn.Module):
    def __init__(self, vec_dim, local_shape, global_shape, feat=64, hidden=256):
        super().__init__()
        self.local = _cnn(*local_shape, feat)
        self.glob = _cnn(*global_shape, feat)
        self.vec = nn.Sequential(nn.Linear(vec_dim, feat), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(3 * feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, obs):
        return self.mlp(torch.cat([self.local(obs["local"]), self.glob(obs["global"]), self.vec(obs["vec"])], 1))


class ActorCritic(nn.Module):
    def __init__(self, vec_dim=7, local_shape=(3, 16, 16), global_shape=(4, 16, 16), log_std_init=-0.7):
        super().__init__()
        self.shapes = dict(vec_dim=vec_dim, local_shape=tuple(local_shape), global_shape=tuple(global_shape))
        self.actor = Encoder(vec_dim, local_shape, global_shape)
        self.critic = Encoder(vec_dim, local_shape, global_shape)
        self.mu = nn.Linear(256, 2)
        self.v = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.full((2,), float(log_std_init)))
        nn.init.orthogonal_(self.mu.weight, 0.01)
        nn.init.zeros_(self.mu.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)
        nn.init.zeros_(self.v.bias)

    def mean(self, obs):
        return torch.tanh(self.mu(self.actor(obs)))

    def value(self, obs):
        return self.v(self.critic(obs)).squeeze(-1)

    def dist(self, obs):
        return torch.distributions.Normal(self.mean(obs), self.log_std.exp())


def save_checkpoint(path, model, obs_cfg, extra=None):
    torch.save(dict(state_dict=model.state_dict(), shapes=model.shapes, obs_cfg=obs_cfg, extra=extra or {}), path)
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(obs_cfg, f, indent=2)


class PPOPolicy:
    """SB3-like `.predict(obs)` over a trained checkpoint; returns thrust in the reference
    ship's units (|u_i| <= u_max), as `ShipEnv` and `benchmark_dp.py` expect."""

    def __init__(self, path, device="cpu"):
        from dynamics import ShipParams
        ck = torch.load(path, map_location=device, weights_only=False)
        self.model = ActorCritic(**ck["shapes"]).to(device).eval()
        self.model.load_state_dict(ck["state_dict"])
        self.obs_cfg = ck["obs_cfg"]
        self.device = device
        self.u_max = ShipParams().u_max

    def predict(self, obs, deterministic=True):
        single = obs["vec"].ndim == 1
        t = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=self.device) for k, v in obs.items()}
        if single:
            t = {k: v[None] for k, v in t.items()}
        with torch.no_grad():
            a = self.model.mean(t) if deterministic else self.model.dist(t).sample().clamp(-1, 1)
        a = a.cpu().numpy() * self.u_max
        return (a[0] if single else a), None
