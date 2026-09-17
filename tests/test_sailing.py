"""Tests for the sailing package: polar, geometry, environment, isochrone and DP planners."""

import os
import sys
import warnings

import numpy as np
import pytest

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, REPO_DIR)

from wind import uniform_wind_field, generate_wind_field  # noqa: E402
from sailing.polar import Polar, wind_geometry, maneuver_kind  # noqa: E402
from sailing.boat_env import SailParams, SailEnv, rollout  # noqa: E402
from sailing.isochrone import isochrone_route, route_follower  # noqa: E402
from sailing.dp_sail import SailDP  # noqa: E402

NORTHERLY_12KT = dict(wx=0.0, wy=-6.0)   # air moving south, 6 units * 2 kt/unit


@pytest.fixture(scope="module")
def polar():
    return Polar.synthetic()


# ------------------------------------------------------------------- polar
def test_polar_shape(polar):
    for tws in (6.0, 12.0, 20.0):
        assert polar.speed(30.0, tws) == 0.0                              # no-go zone
        assert polar.speed(90.0, tws) > polar.speed(50.0, tws) > 0.0      # beam faster than close-hauled
        assert np.isclose(polar.speed(70.0, tws), polar.speed(-70.0, tws))  # symmetric
    assert polar.speed(90.0, 20.0) > polar.speed(90.0, 12.0) > polar.speed(90.0, 6.0)
    up, dn = polar.best_vmg(12.0, True), polar.best_vmg(12.0, False)
    assert 40.0 <= up[0] <= 55.0
    assert 140.0 <= dn[0] < 180.0          # gybing downwind beats running dead downwind


def test_polar_pol_roundtrip(polar, tmp_path):
    path = tmp_path / "boat.pol"
    polar.save_pol(path)
    back = Polar.from_pol(path)
    for twa, tws in ((45, 8), (90, 12), (150, 16), (175, 20)):
        assert abs(back.speed(twa, tws) - polar.speed(twa, tws)) < 0.15


# ---------------------------------------------------------------- geometry
def test_wind_geometry_angles():
    for heading, twa in ((np.pi / 2, 0.0), (-np.pi / 2, 180.0), (0.0, 90.0), (np.pi, 90.0)):
        tws, a, _ = wind_geometry(0.0, -6.0, heading, 2.0)
        assert np.isclose(tws, 12.0) and np.isclose(a, twa, atol=1e-9)


def test_maneuver_kind_rules():
    d = np.deg2rad
    assert maneuver_kind(np.array(d(-45)), np.array(d(45))) == 1     # close-hauled switch: tack
    assert maneuver_kind(np.array(d(-150)), np.array(d(150))) == 2   # both legs downwind: gybe
    assert maneuver_kind(np.array(d(-50)), np.array(d(135))) == 1    # chicken gybe charged as a tack
    assert maneuver_kind(np.array(d(40)), np.array(d(60))) == 0      # same side


# --------------------------------------------------------------------- env
def test_env_beam_reach_time(polar):
    p = SailParams()
    wf = uniform_wind_field(**NORTHERLY_12KT)
    env = SailEnv(wf, polar, p)
    res = rollout(env, lambda e: 0.0, np.array([1.0, 5.0]), np.array([9.0, 5.0]))
    v_units = polar.speed(90.0, 12.0) / p.nm_per_unit
    expect = (8.0 - p.goal_radius) / v_units
    assert res["success"] and res["tacks"] == 0
    assert abs(res["t"] - expect) < 1e-6


def test_env_tack_costs_time(polar):
    p = SailParams()
    wf = uniform_wind_field(**NORTHERLY_12KT)
    env = SailEnv(wf, polar, p)
    env.reset(options=dict(start=(5.0, 1.0), goal=(5.0, 9.0), heading=np.deg2rad(45)))
    env.step(np.array([np.deg2rad(45)]))
    y0 = env.state[1]
    env.step(np.array([np.deg2rad(135)]))          # tack
    assert env.tacks == 1
    assert np.isclose(env.pending, max(0.0, p.tack_time - p.dt))
    assert env.state[1] - y0 < 1e-12 if p.tack_time >= p.dt else True   # no progress while tacking


# ---------------------------------------------------------------- planners
def _upwind_bound(polar, p, n_headings):
    """Best heading-discretised VMG time for the 80 nm beat, one tack, 2.5 nm goal disc."""
    angles = np.rad2deg(np.linspace(-np.pi, np.pi, n_headings, endpoint=False) - np.pi / 2)
    twa = np.abs((angles + 180) % 360 - 180)
    vmg = max(polar.speed(a, 12.0) * np.cos(np.deg2rad(a)) for a in twa if a < 90)
    return (80.0 - p.goal_radius * p.nm_per_unit) / vmg + p.tack_time


def test_dp_upwind_beat_near_optimal(polar):
    p = SailParams()
    wf = uniform_wind_field(**NORTHERLY_12KT)
    s, g = np.array([5.0, 1.0]), np.array([5.0, 9.0])
    dp = SailDP(wf, polar, p, g, nx=61, ny=61, n_headings=36, device="cpu")
    assert dp.solve()["converged"]
    res = rollout(SailEnv(wf, polar, p), dp.policy(), s, g)
    bound = _upwind_bound(polar, p, 36)
    assert res["success"] and res["tacks"] >= 1
    assert bound * 0.97 <= res["t"] <= bound * 1.05, (res["t"], bound)


def test_isochrone_beam_reach_exact(polar):
    p = SailParams()
    wf = uniform_wind_field(**NORTHERLY_12KT)
    s, g = np.array([1.0, 5.0]), np.array([9.0, 5.0])
    iso = isochrone_route(polar, p, s, g, wind=wf, step_h=0.1)
    expect = (8.0 - p.goal_radius) / (polar.speed(90.0, 12.0) / p.nm_per_unit)
    assert iso["success"] and abs(iso["time_h"] - expect) / expect < 0.01


def test_isochrone_upwind_needs_a_tack(polar):
    p = SailParams()
    wf = uniform_wind_field(**NORTHERLY_12KT)
    s, g = np.array([5.0, 1.0]), np.array([5.0, 9.0])
    iso = isochrone_route(polar, p, s, g, wind=wf, step_h=0.1)
    res = rollout(SailEnv(wf, polar, p), route_follower(iso, polar, p), s, g)
    assert iso["success"] and res["success"] and res["tacks"] >= 1
    assert res["t"] <= _upwind_bound(polar, p, 72) * 1.06


def test_isochrone_warns_on_long_penalty(polar):
    p = SailParams(tack_time=0.5)
    wf = uniform_wind_field(**NORTHERLY_12KT)
    with pytest.warns(UserWarning):
        isochrone_route(polar, p, np.array([1.0, 5.0]), np.array([4.0, 5.0]), wind=wf, step_h=0.1)


def test_planners_agree_on_random_field(polar):
    """Isochrone and DP routes on a random wind field, both executed in the same env."""
    p = SailParams()
    wf = generate_wind_field(1)
    s, g = np.array([1.5, 1.0]), np.array([8.0, 8.5])
    dp = SailDP(wf, polar, p, g, nx=61, ny=61, device="cpu")
    dp.solve()
    d = rollout(SailEnv(wf, polar, p), dp.policy(), s, g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iso = isochrone_route(polar, p, s, g, wind=wf, step_h=0.1, n_sectors=720)
    i = rollout(SailEnv(wf, polar, p), route_follower(iso, polar, p), s, g)
    assert d["success"] and i["success"]
    assert abs(i["t"] - d["t"]) / d["t"] < 0.10, (i["t"], d["t"])
