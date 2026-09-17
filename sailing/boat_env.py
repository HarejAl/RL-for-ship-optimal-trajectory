"""
Sailing-yacht environment.

Kinematic boat: its velocity is the polar speed along the commanded heading, evaluated with
the true wind at its position. There is no inertia, so the only thing that makes changing
side expensive is the explicit manoeuvre penalty -- a tack or gybe costs `tack_time` /
`gybe_time` hours of no progress. Without that penalty the optimal route would zig-zag at
every step, since a sailboat can approximate any heading inside the no-go zone by switching
tacks infinitely fast.

Observation : [x, y, cos h, sin h, xg, yg]    (float32; h = current heading)
Action      : absolute heading, radians, shape (1,)
Reward      : -dt per step (minimum time), + goal_bonus on arrival, - oob_penalty on leaving
Episode     : ends inside the goal disc, outside the domain, or at max_steps

The planners in this package use exactly the same polar, geometry and penalties, so a route
from any of them can be executed here and its time compared on equal terms.
"""

from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sailing.polar import Polar, wind_geometry, maneuver_kind


@dataclass(frozen=True)
class SailParams:
    nm_per_unit: float = 10.0        # 1 domain unit = 10 nautical miles
    kts_per_wind_unit: float = 2.0   # WindField units -> knots (synthetic fields: 10 units = 20 kt)
    dt: float = 0.05                 # hours per env step (3 min)
    # Strategic manoeuvre penalties, as routers use them. A crew tack physically costs ~15-60 s,
    # but with a penalty that small the planners' numerical noise dominates it and routes
    # chatter (12-13 tacks on a beat where one is optimal). Configurable.
    tack_time: float = 0.1           # hours of lost progress per tack (6 min)
    gybe_time: float = 0.05          # hours per gybe (3 min)
    goal_radius: float = 0.25        # domain units (2.5 nm)
    min_speed_kts: float = 0.3       # below this the boat is considered stopped

    def kts_to_units_per_h(self, kts):
        return kts / self.nm_per_unit


def boat_velocity(polar, p, wx, wy, heading):
    """Velocity (units/h) along `heading`, plus (speed_kts, rel). numpy, broadcastable."""
    tws, twa, rel = wind_geometry(wx, wy, heading, p.kts_per_wind_unit)
    kts = polar.speed(twa, tws)
    v = p.kts_to_units_per_h(kts)
    return v * np.cos(heading), v * np.sin(heading), kts, rel


class SailEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, wind=None, polar=None, params=None, wind_fn=None, spawn_box=(0.0, 10.0),
                 min_start_goal_dist=4.0, max_steps=2000, goal_bonus=10.0, oob_penalty=10.0):
        """
        wind    : a static WindField, or
        wind_fn : callable(t_hours) -> WindField for time-varying weather
        """
        super().__init__()
        if wind is None and wind_fn is None:
            raise ValueError("provide wind or wind_fn")
        self.wind = wind
        self.wind_fn = wind_fn
        self.polar = polar or Polar.synthetic()
        self.p = params or SailParams()
        self.spawn_box = spawn_box
        self.min_start_goal_dist = min_start_goal_dist
        self.max_steps = max_steps
        self.goal_bonus = goal_bonus
        self.oob_penalty = oob_penalty
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.state = np.zeros(3)  # x, y, heading
        self.goal = np.zeros(2)

    # ----------------------------------------------------------------- helpers
    def wind_at(self, t):
        return self.wind_fn(t) if self.wind_fn is not None else self.wind

    @property
    def extent(self):
        return self.wind_at(0.0).extent

    def _obs(self):
        x, y, h = self.state
        return np.array([x, y, np.cos(h), np.sin(h), *self.goal], dtype=np.float32)

    def _info(self, **kw):
        info = dict(t=self.t, success=False, oob=False, tacks=self.tacks, gybes=self.gybes,
                    dist=float(np.hypot(*(self.state[:2] - self.goal))))
        info.update(kw)
        return info

    # ----------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "wind" in options:
            self.wind = options["wind"]
        lo, hi = self.spawn_box
        start = np.asarray(options["start"], float) if "start" in options else self.np_random.uniform(lo, hi, 2)
        if "goal" in options:
            goal = np.asarray(options["goal"], float)
        else:
            goal = self.np_random.uniform(lo, hi, 2)
            for _ in range(200):
                if np.linalg.norm(goal - start) >= self.min_start_goal_dist:
                    break
                goal = self.np_random.uniform(lo, hi, 2)
        h0 = float(options.get("heading", np.arctan2(*(goal - start)[::-1])))
        self.state = np.array([start[0], start[1], h0])
        self.goal = goal
        self.t = 0.0
        self.steps = 0
        self.pending = 0.0        # manoeuvre time still to be served (hours)
        self.tacks = 0
        self.gybes = 0
        self._started = bool(options.get("heading_is_set", False))
        return self._obs(), self._info()

    def step(self, action):
        p = self.p
        x, y, h_old = self.state
        h_new = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        field = self.wind_at(self.t)
        wx, wy = field(x, y)
        wx, wy = float(wx), float(wy)

        if self._started:
            _, _, rel_old = wind_geometry(wx, wy, h_old, p.kts_per_wind_unit)
            _, _, rel_new = wind_geometry(wx, wy, h_new, p.kts_per_wind_unit)
            kind = int(maneuver_kind(np.asarray(rel_old), np.asarray(rel_new)))
            if kind == 1:
                self.pending += p.tack_time
                self.tacks += 1
            elif kind == 2:
                self.pending += p.gybe_time
                self.gybes += 1
        self._started = True

        move = max(0.0, p.dt - self.pending)
        self.pending = max(0.0, self.pending - p.dt)
        vx, vy, kts, _ = boat_velocity(self.polar, p, wx, wy, h_new)

        x1, y1 = x + vx * move, y + vy * move
        # arrival: first moment along the segment that enters the goal disc
        arrived, tau = _segment_hits_disc(x, y, vx, vy, move, self.goal, p.goal_radius)
        elapsed = ((p.dt - move) + tau) if arrived else p.dt
        if arrived:
            x1, y1 = x + vx * tau, y + vy * tau
        self.t += elapsed
        self.state = np.array([x1, y1, h_new])
        self.steps += 1

        reward = -elapsed   # minimum time
        terminated = truncated = False
        oob = False
        if arrived:
            terminated = True
            reward += self.goal_bonus
        else:
            xmin, xmax, ymin, ymax = field.extent
            if not (xmin <= x1 <= xmax and ymin <= y1 <= ymax):
                terminated = oob = True
                reward -= self.oob_penalty
            elif self.steps >= self.max_steps:
                truncated = True
        return self._obs(), float(reward), terminated, truncated, self._info(
            success=arrived, oob=oob, speed_kts=float(kts))


def _segment_hits_disc(x, y, vx, vy, T, goal, R):
    """Earliest tau in [0, T] with |(x,y) + v*tau - goal| <= R. Returns (hit, tau)."""
    px, py = x - goal[0], y - goal[1]
    if px * px + py * py <= R * R:
        return True, 0.0
    a = vx * vx + vy * vy
    if a <= 1e-15 or T <= 0:
        return False, 0.0
    b = 2.0 * (px * vx + py * vy)
    c = px * px + py * py - R * R
    disc = b * b - 4 * a * c
    if disc < 0:
        return False, 0.0
    tau = (-b - np.sqrt(disc)) / (2 * a)
    return (0.0 <= tau <= T), float(max(tau, 0.0))


def rollout(env, policy, start, goal, wind=None, heading=None, max_steps=None):
    """Run `policy(env) -> heading` from start to goal. Returns trajectory, time, outcome."""
    opts = dict(start=start, goal=goal)
    if wind is not None:
        opts["wind"] = wind
    if heading is not None:
        opts["heading"] = heading
    env.reset(options=opts)
    traj = [env.state.copy()]
    info = env._info()
    for _ in range(max_steps or env.max_steps):
        _, _, term, trunc, info = env.step(np.array([policy(env)]))
        traj.append(env.state.copy())
        if term or trunc:
            break
    return dict(traj=np.array(traj), t=info["t"], success=info["success"], oob=info["oob"],
                tacks=info["tacks"], gybes=info["gybes"])
