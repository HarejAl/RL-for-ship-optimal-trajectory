"""Unit and sanity tests for wind fields, dynamics, environment and the DP baseline."""

import os
import sys

import numpy as np
import pytest
import torch

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, REPO_DIR)

from dynamics import ShipParams, ship_step, terminal_speed  # noqa: E402
from wind import WindField, generate_wind_field, uniform_wind_field  # noqa: E402
from env import ShipEnv  # noqa: E402
from dp_baseline import ValueIterationPlanner  # noqa: E402

LEGACY = os.path.join(REPO_DIR, "WF.pkl")


# ------------------------------------------------------------------- wind
def test_legacy_roundtrip():
    wf = WindField.load_legacy(LEGACY)
    d = wf.to_legacy_dict()
    import pickle
    with open(LEGACY, "rb") as f:
        orig = pickle.load(f)
    assert np.allclose(d["Intensity"], orig["Intensity"], atol=1e-9)
    mask = orig["Intensity"] > 1e-6  # direction undefined where speed is zero
    dd = np.mod(d["Direction"] - orig["Direction"] + np.pi, 2 * np.pi) - np.pi
    assert np.allclose(dd[mask], 0.0, atol=1e-6)


def test_interp_numpy_matches_torch():
    wf = generate_wind_field(0)
    rng = np.random.default_rng(1)
    px = rng.uniform(-3, 13, 500)  # includes out-of-grid points (edge clamped)
    py = rng.uniform(-3, 13, 500)
    wx, wy = wf(px, py)
    sample = wf.torch_sampler("cpu", torch.float64)
    twx, twy = sample(torch.as_tensor(px), torch.as_tensor(py))
    assert np.allclose(wx, twx.numpy(), atol=1e-10)
    assert np.allclose(wy, twy.numpy(), atol=1e-10)
    # exact on nodes
    wx0, wy0 = wf(wf.x[7], wf.y[3])
    assert np.isclose(wx0, wf.wx[7, 3]) and np.isclose(wy0, wf.wy[7, 3])


def test_generator_shape_cap_and_seed():
    a = generate_wind_field(42, nx=51, ny=61, max_speed=8.0)
    b = generate_wind_field(42, nx=51, ny=61, max_speed=8.0)
    assert a.wx.shape == (51, 61)
    assert a.speed.max() <= 8.0 + 1e-9
    assert np.array_equal(a.wx, b.wx) and np.array_equal(a.wy, b.wy)
    c = generate_wind_field(43, nx=51, ny=61, max_speed=8.0)
    assert not np.allclose(a.wx, c.wx)


# --------------------------------------------------------------- dynamics
def test_step_numpy_equals_torch():
    p = ShipParams()
    rng = np.random.default_rng(0)
    s = rng.uniform(-5, 5, (100, 4))
    u = rng.uniform(-10, 10, (100, 2))
    w = rng.uniform(-10, 10, (100, 2))
    out_np = ship_step(s[:, 0], s[:, 1], s[:, 2], s[:, 3], u[:, 0], u[:, 1], w[:, 0], w[:, 1], p)
    T = lambda a: torch.as_tensor(a, dtype=torch.float64)
    out_t = ship_step(T(s[:, 0]), T(s[:, 1]), T(s[:, 2]), T(s[:, 3]),
                      T(u[:, 0]), T(u[:, 1]), T(w[:, 0]), T(w[:, 1]), p)
    for a, b in zip(out_np, out_t):
        assert np.allclose(a, b.numpy(), atol=1e-12)


def test_terminal_speed_is_steady_state():
    p = ShipParams()
    w = 6.0
    v = terminal_speed(p, w)
    # full diagonal thrust u_eff along +x, tailwind w along +x: acceleration must vanish at v
    from dynamics import ship_accel
    ax, _ = ship_accel(v, 0.0, p.u_max * 2 ** 0.5, 0.0, w, 0.0, p)
    assert abs(ax) < 1e-9


# -------------------------------------------------------------------- env
def test_env_reset_is_seeded():
    wf = generate_wind_field(0)
    e1, e2 = ShipEnv(wf), ShipEnv(wf)
    o1, _ = e1.reset(seed=3)
    o2, _ = e2.reset(seed=3)
    assert np.array_equal(o1, o2)
    assert np.linalg.norm(o1[:2] - o1[4:]) >= e1.min_start_goal_dist
    a = np.array([3.0, -2.0])
    s1 = e1.step(a)[0]
    s2 = e2.step(a)[0]
    assert np.array_equal(s1, s2)
    o3, _ = e1.reset(seed=4)
    assert not np.array_equal(o1, o3)


def test_env_reaches_goal_and_tracks_cost():
    env = ShipEnv(uniform_wind_field())
    env.reset(seed=0, options=dict(start=(2.0, 2.0), goal=(2.0, 5.0)))
    J = 0.0
    for k in range(600):
        obs, r, term, trunc, info = env.step(np.array([0.0, 10.0]))
        J += info["stage_cost"]
        if term:
            break
    assert info["success"] and not info["oob"]
    assert np.isclose(info["J"], J)
    assert np.isclose(info["t"], (k + 1) * env.p.dt)


def test_env_out_of_bounds_terminates():
    env = ShipEnv(uniform_wind_field())
    env.reset(seed=0, options=dict(start=(0.5, 5.0), goal=(9.0, 5.0)))
    for _ in range(600):
        obs, r, term, trunc, info = env.step(np.array([-10.0, 0.0]))
        if term:
            break
    assert info["oob"] and not info["success"]


# --------------------------------------------------------------- baseline
def _small_planner(wind, goal, **kw):
    args = dict(nx=31, ny=31, nv=9, n_act=5, exec_n_act=7, device="cpu", verbose=False)
    args.update(kw)
    return ValueIterationPlanner(wind, goal, **args)


def test_dp_zero_wind_goes_straight():
    wf = uniform_wind_field()
    env = ShipEnv(wf)
    start, goal = np.array([2.0, 2.0]), np.array([7.0, 7.0])
    vip = _small_planner(wf, goal)
    stats = vip.solve(max_iter=800, tol=1e-3, verbose=False)
    assert stats["converged"]
    res = vip.rollout(env, start)
    assert res["success"], res
    # crossing 7.07 units at terminal speed ~5.3 takes ~1.4 s; allow a generous margin
    assert res["J"] < 8.0, res["J"]
    # path stays close to the straight line
    d = start - goal
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    dev = np.abs((res["traj"][:, :2] - goal) @ n)
    assert dev.max() < 0.5, dev.max()
    # value is monotone: further from the goal costs more
    assert vip.value(2.0, 2.0) > vip.value(5.0, 5.0) > vip.value(6.9, 6.9)


def test_dp_tailwind_cheaper_than_headwind():
    wf = uniform_wind_field(wx=5.0)
    env = ShipEnv(wf)
    tail = _small_planner(wf, (9.0, 5.0))
    tail.solve(max_iter=800, tol=1e-3, verbose=False)
    r_tail = tail.rollout(env, (1.0, 5.0))
    head = _small_planner(wf, (1.0, 5.0))
    head.solve(max_iter=800, tol=1e-3, verbose=False)
    r_head = head.rollout(env, (9.0, 5.0))
    assert r_tail["success"] and r_head["success"]
    assert r_tail["J"] < r_head["J"]
    assert r_tail["t"] < r_head["t"]


# ------------------------------------------------------------ wind-aware obs
def test_wind_obs_wrapper_shapes_and_content():
    from wind_obs import WindObsWrapper, WindFieldPool, make_wind_env
    pool = WindFieldPool(3, seed_base=5_000_000, nx=41, ny=41)
    env = make_wind_env(pool=pool, obs_cfg=dict(local_res=8, global_res=12), monitor=False)
    obs, _ = env.reset(seed=1)
    assert set(obs) == {"vec", "local", "global"}
    assert obs["local"].shape == (3, 8, 8) and obs["global"].shape == (4, 12, 12)
    assert env.observation_space.contains(obs)
    base = env.unwrapped
    x, y = base.state[:2]
    # centre of the local crop equals the wind at the ship (K even: average of the 4 centre pixels)
    wx, wy = base.wind(x, y)
    c = obs["local"][0, 3:5, 3:5].mean() * 10
    assert abs(c - wx) < 0.5
    # ship blob peaks near the ship, goal blob near the goal
    gi = np.unravel_index(obs["global"][2].argmax(), obs["global"][2].shape)
    xmin, xmax, _, _ = base.wind.extent
    cell = (xmax - xmin) / 11
    assert abs(xmin + gi[0] * cell - x) <= cell and abs(xmin + gi[1] * cell - y) <= cell
    # wind field changes between resets (pool of 3, seeded)
    fields = {id(env.unwrapped.wind) for _ in range(10) if env.reset()[0] is not None}
    assert len(fields) > 1
    # step keeps the dict shape
    obs2, r, term, trunc, info = env.step(np.array([1.0, 1.0]))
    assert env.observation_space.contains(obs2)


def test_cnn_extractor_forward():
    import torch
    from wind_obs import WindCNNExtractor, WindFieldPool, make_wind_env
    pool = WindFieldPool(1, seed_base=5_000_000, nx=41, ny=41)
    env = make_wind_env(pool=pool, monitor=False)
    ext = WindCNNExtractor(env.observation_space)
    obs, _ = env.reset(seed=0)
    batch = {k: torch.as_tensor(v)[None] for k, v in obs.items()}
    out = ext(batch)
    assert out.shape == (1, ext.features_dim) and ext.features_dim == 64 * 3


def test_goal_radius_curriculum_shrinks_with_success():
    env = ShipEnv(uniform_wind_field(), goal_radius=0.25, curriculum=(1.0, 0.5, 0.5, 4))
    assert env.current_radius == 1.0
    for _ in range(4):  # four successes in a row -> shrink
        env.reset(seed=0, options=dict(start=(2.0, 2.0), goal=(2.0, 4.0)))
        for _ in range(600):
            _, _, term, trunc, info = env.step(np.array([0.0, 10.0]))
            if term or trunc:
                break
        assert info["success"]
    assert env.current_radius == 0.5
    for _ in range(8):
        env.reset(seed=0, options=dict(start=(2.0, 2.0), goal=(2.0, 4.0)))
        for _ in range(600):
            _, _, term, trunc, info = env.step(np.array([0.0, 10.0]))
            if term or trunc:
                break
    assert env.current_radius == 0.25  # never below goal_radius
