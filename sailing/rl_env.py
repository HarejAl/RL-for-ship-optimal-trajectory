"""
RL-facing wrapper for `SailEnv`: goal-aligned observations, relative actions, shaped reward.

Everything is expressed in the frame where the goal lies straight ahead (+u axis). The same
situation -- say, goal dead upwind at 50 nm in 12 kt -- then looks identical whatever the
compass direction, so the policy does not have to relearn tacking for every wind direction.

Action      a in [-1, 1]  ->  heading = bearing_to_goal + pi * a
            (a = 0 steers at the goal; the +-pi seam is "sail directly away", never wanted)
            or, with n_actions=N, a discrete choice k -> a = -1 + 2k/N

Observation (flat Box, or Dict for the CNN when `map_res` is set)
    vec : dist/10, wind at the boat rotated into the goal frame (/10),
          cos/sin(heading - bearing), manoeuvre time still pending / tack_time,
          cos/sin(bearing), x/10 - 0.5, y/10 - 0.5          (last four: where the walls are)
    map : (4, R, R) egocentric grid, u in [-3, 9], v in [-6, 6], goal-aligned:
          wind u/v (/10), inside-domain mask, goal blob

Reward      env reward (-hours elapsed, +10 on arrival, -10 off the map)
            + potential shaping  phi(s') - phi(s), phi = -(hours still needed), with
            shaping="dist" : dist / 6 kt              (plain "closer is better")
            shaping="time" : dist / best speed made good to the goal the polar allows in the
                             local wind -- knows about the no-go zone, and rates a position in
                             stronger wind as closer in TIME even when it is further in distance
            shaping="none" : no shaping at all, pure minimum time

Scenario samplers draw (wind, start, goal) per episode:
    UniformWindScenarios : constant wind, random direction and strength; a share of episodes
                           forces the goal upwind (must tack) or downwind (should gybe)
    FieldPoolScenarios   : random spatially varying fields from `wind.generate_wind_field`
    RegattaScenarios     : a band of stronger breeze (or a lull) beside the rhumb line + gusts,
                           half the courses beats -- the map says which side pays
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from wind import generate_wind_field, uniform_wind_field
from sailing.boat_env import SailEnv, SailParams
from sailing.polar import Polar, wind_geometry

RL_PARAMS = SailParams(dt=0.1)      # 6-min decisions: a 50 nm leg is ~100 steps
V_REF_KTS = 6.0
TRAIN_SEED_BASE = 3_000_000          # sailing field seeds, disjoint from the ship project's
EVAL_SEED_BASE = 4_000_000


def _rot(x, y, c, s):
    """Rotate vectors by -bearing (c = cos b, s = sin b): world -> goal frame."""
    return c * x + s * y, -s * x + c * y


def _sample_pair(rng, lo=0.0, hi=10.0, min_dist=4.0):
    start = rng.uniform(lo, hi, 2)
    for _ in range(200):
        goal = rng.uniform(lo, hi, 2)
        if np.linalg.norm(goal - start) >= min_dist:
            break
    return start, goal


class UniformWindScenarios:
    def __init__(self, speed_units=(3.0, 9.0), p_upwind=0.35, p_downwind=0.15, cone_deg=25.0):
        self.speed_units, self.p_up, self.p_dn = speed_units, p_upwind, p_downwind
        self.cone = np.deg2rad(cone_deg)

    def __call__(self, rng):
        start, goal = _sample_pair(rng)
        bearing = np.arctan2(*(goal - start)[::-1])
        u = rng.uniform()
        jitter = rng.uniform(-self.cone, self.cone)
        if u < self.p_up:                        # air moves from goal to start: a beat
            blow = bearing + np.pi + jitter
        elif u < self.p_up + self.p_dn:          # air moves from start to goal: a run
            blow = bearing + jitter
        else:
            blow = rng.uniform(-np.pi, np.pi)
        spd = rng.uniform(*self.speed_units)
        wind = uniform_wind_field(spd * np.cos(blow), spd * np.sin(blow), nx=5, ny=5)
        return wind, start, goal


class FieldPoolScenarios:
    def __init__(self, n_fields=500, seed_base=TRAIN_SEED_BASE):
        self.seeds = [seed_base + i for i in range(n_fields)]
        self._cache = {}

    def field(self, seed):
        if seed not in self._cache:
            self._cache[seed] = generate_wind_field(seed)
        return self._cache[seed]

    def __call__(self, rng):
        seed = self.seeds[int(rng.integers(len(self.seeds)))]
        start, goal = _sample_pair(rng)
        return self.field(seed), start, goal


class RegattaScenarios:
    """
    Winds with structure worth reading. A fresh field per episode:
      * base wind of random direction, 8-16 kt
      * a band of stronger breeze (or a lull) parallel to the wind, placed 1-3 units to one side
        of the rhumb line -- so on a beat one side of the course pays and the other does not
      * smooth gust patches (+-12 %) and a direction wobble (+-8 deg)
    Half the courses are beats, where choosing the side is the whole game.
    """

    def __init__(self, p_upwind=0.5, p_downwind=0.15, cone_deg=25.0, n=61, extent=(-1.0, 11.0)):
        self.p_up, self.p_dn, self.cone = p_upwind, p_downwind, np.deg2rad(cone_deg)
        self.x = np.linspace(*extent, n)
        self.X, self.Y = np.meshgrid(self.x, self.x, indexing="ij")
        self.dx = self.x[1] - self.x[0]

    def __call__(self, rng):
        from scipy.ndimage import gaussian_filter
        from wind import WindField
        start, goal = _sample_pair(rng)
        bearing = np.arctan2(*(goal - start)[::-1])
        u = rng.uniform()
        jitter = rng.uniform(-self.cone, self.cone)
        if u < self.p_up:
            theta = bearing + np.pi + jitter
        elif u < self.p_up + self.p_dn:
            theta = bearing + jitter
        else:
            theta = rng.uniform(-np.pi, np.pi)
        base = rng.uniform(8.0, 16.0) / RL_PARAMS.kts_per_wind_unit
        dvec = np.array([np.cos(theta), np.sin(theta)])
        nvec = np.array([-dvec[1], dvec[0]])
        mid = 0.5 * (start + goal)
        centre = mid + nvec * rng.uniform(1.0, 3.0) * rng.choice([-1.0, 1.0]) + dvec * rng.uniform(-2, 2)
        dist = (self.X - centre[0]) * nvec[0] + (self.Y - centre[1]) * nvec[1]
        width = rng.uniform(1.2, 3.0)
        strength = rng.uniform(0.3, 0.6) if rng.uniform() < 0.75 else -rng.uniform(0.3, 0.5)
        pressure = 1.0 + strength * np.exp(-dist ** 2 / (2 * width ** 2))
        g = [gaussian_filter(rng.standard_normal(self.X.shape), 1.5 / self.dx, mode="reflect") for _ in range(2)]
        g = [a / a.std() for a in g]
        ang = theta + np.deg2rad(8.0) * g[0]
        speed = np.maximum(base * pressure * (1.0 + 0.12 * g[1]), 0.0)
        wind = WindField(self.x, self.x, speed * np.cos(ang), speed * np.sin(ang))
        return wind, start, goal


class SailRLEnv(gym.Wrapper):
    def __init__(self, scenarios=None, polar=None, params=RL_PARAMS, map_res=None, gamma=0.995,
                 shaping="dist", max_steps=600, n_actions=None, n_probe=72):
        base = SailEnv(wind=uniform_wind_field(nx=5, ny=5), polar=polar or Polar.synthetic(),
                       params=params, max_steps=max_steps)
        super().__init__(base)
        self.scenarios = scenarios
        self.p = params
        self.gamma = gamma
        self.shaping = {True: "dist", False: "none"}.get(shaping, shaping)
        if self.shaping not in ("dist", "time", "none"):
            raise ValueError("shaping must be 'dist', 'time' or 'none'")
        self.polar = base.polar
        self._probe = np.linspace(-np.pi, np.pi, n_probe, endpoint=False)
        self.map_res = map_res
        self.n_actions = n_actions
        self.action_space = (spaces.Discrete(n_actions) if n_actions
                             else spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32))
        vec = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
        if map_res:
            self.observation_space = spaces.Dict(
                vec=vec, map=spaces.Box(-np.inf, np.inf, shape=(4, map_res, map_res), dtype=np.float32))
            u = np.linspace(-3.0, 9.0, map_res)
            v = np.linspace(-6.0, 6.0, map_res)
            self._U, self._V = np.meshgrid(u, v, indexing="ij")
        else:
            self.observation_space = vec

    # ----------------------------------------------------------------- frame
    @property
    def sail(self):
        return self.env

    def _frame(self):
        x, y, _ = self.sail.state
        dx, dy = self.sail.goal[0] - x, self.sail.goal[1] - y
        b = np.arctan2(dy, dx)
        return b, np.cos(b), np.sin(b), float(np.hypot(dx, dy))

    def _phi(self):
        """Potential: minus an estimate of the hours still needed to reach the goal.

        shaping="dist": distance / a fixed 6 kt. Simple, but it charges the boat for any move
        that increases distance -- including the long board into stronger wind that wins races.

        shaping="time": distance / the best speed made good to the goal that the polar allows in
        the wind here. Upwind that is the tacking VMG; on a reach it is simply the boat speed
        while pointing at the mark; head to wind it is ~0, so the no-go zone looks as bad as it
        is. Sitting in more wind raises the speed and so RAISES the potential: sailing into
        pressure earns reward instead of losing it.
        """
        b, c, s, d = self._frame()
        if self.shaping != "time":
            return -d * self.p.nm_per_unit / V_REF_KTS
        e = self.sail
        x, y, _ = e.state
        wx, wy = e.wind_at(e.t)(x, y)
        tws, twa, _ = wind_geometry(float(wx), float(wy), b + self._probe, self.p.kts_per_wind_unit)
        vmg = float(np.max(self.polar.speed(twa, tws) * np.cos(self._probe)))
        return -d * self.p.nm_per_unit / max(vmg, self.p.min_speed_kts)

    def observe(self):
        e = self.sail
        x, y, h = e.state
        b, c, s, d = self._frame()
        wind = e.wind_at(e.t)
        wx, wy = wind(x, y)
        wu, wv = _rot(float(wx), float(wy), c, s)
        vec = np.array([d / 10.0, wu / 10.0, wv / 10.0, np.cos(h - b), np.sin(h - b),
                        min(e.pending / max(self.p.tack_time, 1e-9), 2.0),
                        c, s, x / 10.0 - 0.5, y / 10.0 - 0.5], dtype=np.float32)
        if not self.map_res:
            return vec
        px = x + c * self._U - s * self._V            # goal frame -> world
        py = y + s * self._U + c * self._V
        mwx, mwy = wind(px, py)
        mu, mv = _rot(mwx, mwy, c, s)
        xmin, xmax, ymin, ymax = wind.extent
        inside = (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)
        goal = np.exp(-((self._U - d) ** 2 + self._V ** 2) / (2 * 0.6 ** 2))
        m = np.stack((mu / 10.0, mv / 10.0, inside, goal)).astype(np.float32)
        return dict(vec=vec, map=m)

    def action_to_a(self, action):
        if self.n_actions:
            return -1.0 + 2.0 * int(np.asarray(action).reshape(-1)[0]) / self.n_actions
        return float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))

    def heading_for(self, a):
        b = self._frame()[0]
        return float(np.arctan2(np.sin(b + np.pi * a), np.cos(b + np.pi * a)))

    # ----------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif not hasattr(self, "_rng"):
            self._rng = np.random.default_rng()
        options = dict(options or {})
        if not {"wind", "start", "goal"} <= options.keys():
            wind, start, goal = self.scenarios(self._rng)
            options.update(wind=wind, start=start, goal=goal)
        self.sail.wind = options["wind"]
        _, info = self.sail.reset(options=options)
        return self.observe(), info

    def step(self, action):
        a = self.action_to_a(action)
        phi0 = self._phi() if self.shaping != "none" else 0.0
        _, r, term, trunc, info = self.sail.step(np.array([self.heading_for(a)]))
        if self.shaping == "none":
            pass
        elif not info["success"]:
            # NOTE: gamma = 1 inside the shaping term on purpose. With gamma < 1 the term
            # gamma*phi - phi = phi*(gamma-1) is a per-step BONUS proportional to how far the
            # goal still is: with the time potential far upwind (phi ~ -30 h) that bonus
            # (+0.15) exceeds the -0.1 h cost of a step, so stalling forever beats finishing.
            r += self._phi() - phi0
        else:
            r += -phi0            # phi(goal) = 0: arriving collects the remaining potential
        return self.observe(), float(r), term, trunc, info


def _scenarios(mode, n_fields, seed_base=TRAIN_SEED_BASE):
    if mode == "uniform":
        return UniformWindScenarios()
    if mode == "regatta":
        return RegattaScenarios()          # a fresh field every episode
    return FieldPoolScenarios(n_fields, seed_base)


def make_rl_env(mode="uniform", n_fields=500, map_res=None, seed=None, monitor=True, **kw):
    scen = _scenarios(mode, n_fields)
    env = SailRLEnv(scen, map_res=map_res, **kw)
    if seed is not None:
        env.reset(seed=seed)
    if monitor:
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env, info_keywords=("success", "t", "tacks", "gybes"))
    return env


def eval_cases(mode="uniform", n=40, seed=12345, n_fields=100):
    """Fixed held-out (wind, start, goal) triples; field seeds never used in training."""
    rng = np.random.default_rng(seed)
    scen = _scenarios(mode, n_fields, EVAL_SEED_BASE)
    return [scen(rng) for _ in range(n)]
