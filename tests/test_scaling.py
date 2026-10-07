"""Ship scales: a faster ship is exactly the reference ship in weaker wind; windage is not."""

import os
import sys

import numpy as np
import pytest

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, REPO_DIR)

from dynamics import ShipParams, ship_step  # noqa: E402
from wind import WindField, generate_wind_field  # noqa: E402
from env import ShipEnv  # noqa: E402
from wind_obs import WindObsWrapper, WindStencilWrapper, wrap_wind_obs, POS_SCALE  # noqa: E402

REF = ShipParams()
# 4x thrust doubles V*; same c_w (same L*) and c_a (same kappa); half the step keeps dt / T*
FAST = ShipParams(u_max=4 * REF.u_max, dt=REF.dt / 2)
K = 2.0   # FAST.scales().speed / REF.scales().speed


def scaled(field, k):
    return WindField(field.x, field.y, k * field.wx, k * field.wy, meta=field.meta)


def simulate(p, wind, thrust_frac, n=300):
    s = np.array([2.0, 2.0, 0.0, 0.0])
    out = [s.copy()]
    for _ in range(n):
        wx, wy = wind(s[0], s[1])
        s = np.array(ship_step(*s, thrust_frac[0] * p.u_max, thrust_frac[1] * p.u_max, wx, wy, p))
        out.append(s)
    return np.array(out)


def test_reference_scales():
    sc = REF.scales()
    assert sc.speed == pytest.approx(np.sqrt(20.0))
    assert sc.length == pytest.approx(2.0)
    assert sc.time == pytest.approx(2.0 / np.sqrt(20.0))
    assert sc.windage == pytest.approx(0.5)
    assert FAST.scales().speed == pytest.approx(K * sc.speed)


def test_faster_ship_is_weaker_wind():
    """Same positions at every step; velocities exactly K times larger."""
    w = generate_wind_field(3)
    a = simulate(REF, w, (0.7, 0.4))
    b = simulate(FAST, scaled(w, K), (0.7, 0.4))
    assert np.allclose(a[:, :2], b[:, :2], atol=1e-9)
    assert np.allclose(K * a[:, 2:], b[:, 2:], atol=1e-9)


def test_windage_is_not_a_wind_rescaling():
    """Halving c_a differs from scaling the wind, even in still air (c_a also damps the ship)."""
    calm = WindField(np.linspace(0, 10, 5), np.linspace(0, 10, 5), np.zeros((5, 5)), np.zeros((5, 5)))
    a = simulate(REF, calm, (0.7, 0.4), n=100)
    b = simulate(ShipParams(cd_air=REF.cd_air / 2), calm, (0.7, 0.4), n=100)
    assert np.abs(a[-1, :2] - b[-1, :2]).max() > 0.5


def _pair(cls, **kw):
    w = generate_wind_field(5)
    ea = cls(ShipEnv(wind=w, params=REF), **kw)
    eb = cls(ShipEnv(wind=scaled(w, K), params=FAST), **kw)
    opts = dict(start=(2.0, 3.0), goal=(8.0, 7.0))
    ea.reset(seed=0, options=opts)
    eb.reset(seed=0, options=opts)
    ea.unwrapped.state = np.array([2.0, 3.0, 1.2, -0.7])
    eb.unwrapped.state = np.array([2.0, 3.0, K * 1.2, -K * 0.7])
    return ea, eb


@pytest.mark.parametrize("cls,kw", [(WindObsWrapper, {}), (WindStencilWrapper, {"n": 3})])
def test_observation_invariant(cls, kw):
    ea, eb = _pair(cls, **kw)
    oa = ea.observation(None)
    ob = eb.observation(None)
    if isinstance(oa, dict):
        for k in oa:
            assert np.allclose(oa[k], ob[k], atol=1e-6), k
    else:
        assert np.allclose(oa, ob, atol=1e-6)


def test_reference_observation_unchanged():
    """Default ship: the same numbers the trained models were fed (vx/6, wind/10)."""
    ea, _ = _pair(WindObsWrapper)
    vec = ea.observation(None)["vec"]
    assert np.allclose(vec[2:4], np.array([1.2, -0.7]) / 6.0)
    assert np.allclose(vec[0:2], np.array([6.0, 4.0]) / POS_SCALE)


def test_trained_policy_transfers_to_faster_ship():
    """bc_t2 on the fast ship in K x wind sails the reference ship's route, at K x the speed."""
    path = os.path.join(REPO_DIR, "models", "bc_t2.zip")
    if not os.path.exists(path):
        pytest.skip("models/bc_t2.zip not available")
    from benchmark_dp import load_model
    model, obs_cfg = load_model(path)
    w = generate_wind_field(11)
    opts = dict(start=(1.5, 1.5), goal=(8.0, 8.0))
    trajs = []
    for p, field in ((REF, w), (FAST, scaled(w, K))):
        env = wrap_wind_obs(ShipEnv(wind=field, params=p), obs_cfg)
        obs, _ = env.reset(seed=0, options=opts)
        xs = [env.unwrapped.state[:2].copy()]
        for _ in range(60):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            xs.append(env.unwrapped.state[:2].copy())
            if term or trunc:
                break
        trajs.append(np.array(xs))
    a, b = trajs
    n = min(len(a), len(b))
    assert np.allclose(a[:n], b[:n], atol=1e-3)


def test_perceived_wind_wrapper():
    """The policy sees the wind scaled by sqrt(kappa/kappa_ref); the dynamics keep the true wind."""
    from wind_obs import PerceivedWindWrapper
    w = generate_wind_field(5)
    opts = dict(start=(2.0, 3.0), goal=(8.0, 7.0))
    ref = WindObsWrapper(ShipEnv(wind=w, params=REF))
    ship = ShipParams(cd_air=0.05)                       # kappa 0.1
    per = WindObsWrapper(PerceivedWindWrapper(ShipEnv(wind=w, params=ship), kappa_ref=0.5))
    o_ref, _ = ref.reset(seed=0, options=opts)
    o_per, _ = per.reset(seed=0, options=opts)
    f = np.sqrt(0.1 / 0.5)
    assert np.allclose(o_per["vec"], o_ref["vec"])
    assert np.allclose(o_per["local"][:2], f * o_ref["local"][:2], atol=1e-6)
    assert np.allclose(o_per["global"][:2], f * o_ref["global"][:2], atol=1e-6)
    per.step(np.array([3.0, 2.0]))
    assert per.env.env.p is ship and per.env.env.wind is w   # true ship and wind drive the step
