"""The batched GPU environment reproduces ShipEnv + WindObsWrapper exactly, in both windage
modes: kappa as a policy input, and the parameter-free perceived wind (PerceivedWindWrapper)."""

import os
import sys

import numpy as np
import pytest
import torch

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from dynamics import ShipParams  # noqa: E402
from env import ShipEnv  # noqa: E402
from wind import WindField, generate_wind_field  # noqa: E402
from wind_obs import WindObsWrapper, PerceivedWindWrapper  # noqa: E402
from gpu_env import FieldBank, BatchShipEnv  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [7, 8, 9]
KAPPA = [0.1, 0.5, 0.3]
MULT = [1.0, 0.4, 0.8]
START = [(1.0, 2.0), (8.0, 1.5), (5.0, 9.0)]
GOAL = [(8.0, 8.0), (2.0, 7.0), (5.0, 1.0)]


MODES = [dict(add_kappa=True), dict(kappa_ref=0.5)]


def cpu_envs(mode):
    out = []
    for s, k, m, st, g in zip(SEEDS, KAPPA, MULT, START, GOAL):
        f = generate_wind_field(s)
        f = WindField(f.x, f.y, m * f.wx, m * f.wy)
        base = ShipEnv(f, params=ShipParams(cd_air=k * 0.5))
        if "kappa_ref" in mode:
            base = PerceivedWindWrapper(base, mode["kappa_ref"])
        e = WindObsWrapper(base, add_kappa=mode.get("add_kappa", False))
        e.reset(seed=0, options=dict(start=st, goal=g))
        out.append(e)
    return out


def gpu_env(mode):
    bank = FieldBank(0, DEVICE, fields=[generate_wind_field(s) for s in SEEDS])
    env = BatchShipEnv(len(SEEDS), bank, **mode)
    env.set_cases(range(len(SEEDS)), START, GOAL, KAPPA, MULT)
    return env


@pytest.mark.parametrize("mode", MODES)
def test_observation_matches_cpu(mode):
    env = gpu_env(mode)
    obs = {k: v.cpu().numpy() for k, v in env.obs().items()}
    for n, e in enumerate(cpu_envs(mode)):
        ref = e.observation(None)
        for k in ref:
            assert np.allclose(obs[k][n], ref[k], atol=1e-5), (n, k)


@pytest.mark.parametrize("mode", MODES)
def test_trajectory_matches_cpu(mode):
    env = gpu_env(mode)
    cpus = cpu_envs(mode)
    rng = np.random.default_rng(0)
    for _ in range(40):
        a = rng.uniform(-1, 1, (len(SEEDS), 2)).astype(np.float32)
        env.step(torch.as_tensor(a, device=env.dev), auto_reset=False)
        for n, e in enumerate(cpus):
            e.step(a[n] * 10.0)
    for n, e in enumerate(cpus):
        base = e.unwrapped
        assert np.allclose(env.pos[n].cpu().numpy(), base.state[:2], atol=1e-3)
        assert np.allclose(env.vel[n].cpu().numpy(), base.state[2:], atol=1e-3)
        assert env.J[n].item() == pytest.approx(base.J, rel=1e-4)


def test_auto_reset_and_shaping():
    env = BatchShipEnv(256, FieldBank(8, DEVICE))
    assert ((env.goal - env.pos).norm(dim=1) >= env.min_dist - 1e-6).all()
    d0 = env.dist().clone()
    _, r, term, trunc, info = env.step(torch.zeros(256, 2, device=env.dev))
    # zero thrust: cost is dt*time_w; the rest of the reward is the potential difference
    expected = -env.p.dt * env.p.time_w + (d0 - env.gamma * env.dist()) / env.v_ref
    assert torch.allclose(r[~info["done"]], expected[~info["done"]], atol=1e-5)
