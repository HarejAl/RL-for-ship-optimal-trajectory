"""
Gymnasium environment: steer a point-mass ship through a 2D wind field to a goal.

Observation : [x, y, vx, vy, xf, yf]          (float32)
Action      : [ux, uy] thrust, each in [-u_max, u_max]
Reward      : -stage_cost(u) + target_w * (d_prev - d)     (potential-based shaping)
              + goal_bonus when the goal disc is entered
              - oob_penalty when the ship leaves the wind grid
Terminal terms are kept at the scale of the dense signal (the shaping sums to the
start distance, 2-10, over an episode); a penalty of 100 buries that structure in
the critic and makes TD3 saturate to a bang-bang exit policy.
Termination : goal reached, or ship outside the wind grid
Truncation  : max_steps

The episode cost J = sum of stage costs (time + control energy) is accumulated in
info["J"], so it can be compared directly with the DP baseline which minimises
the same quantity. The shaping term is potential-based (potential = -target_w*d)
and therefore does not change the ranking of successful trajectories.

Dynamics live in `dynamics.py` and are shared with the baseline.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from dynamics import ShipParams, ship_step, stage_cost
from wind import WindField


class ShipEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        wind=None,
        params=None,
        wind_sampler=None,
        spawn_box=(0.0, 10.0),
        goal_radius=0.25,
        min_start_goal_dist=2.0,
        max_steps=600,
        target_w=1.0,
        goal_bonus=10.0,
        oob_penalty=10.0,
    ):
        """
        wind         : WindField, or a legacy dict with keys x, y, Intensity, Direction
        wind_sampler : optional callable(np_random) -> WindField, resampled at each reset
                       (used to train wind-aware policies on a distribution of fields)
        spawn_box    : start and goal are sampled uniformly in this square
        """
        super().__init__()
        if isinstance(wind, dict):
            wind = WindField.from_legacy_dict(wind)
        if wind is None and wind_sampler is None:
            raise ValueError("provide a wind field or a wind_sampler")
        self.wind = wind
        self.wind_sampler = wind_sampler
        self.p = params or ShipParams()
        self.spawn_box = spawn_box
        self.goal_radius = goal_radius
        self.min_start_goal_dist = min_start_goal_dist
        self.max_steps = max_steps
        self.target_w = target_w
        self.goal_bonus = goal_bonus
        self.oob_penalty = oob_penalty

        self.action_space = spaces.Box(
            low=-self.p.u_max, high=self.p.u_max, shape=(2,), dtype=np.float32
        )
        v_lim = 20.0
        lo = np.array([-20.0, -20.0, -v_lim, -v_lim, -20.0, -20.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=lo, high=-lo, dtype=np.float32)

        self.state = np.zeros(4)
        self.goal = np.zeros(2)
        self.steps = 0
        self.J = 0.0

    # ------------------------------------------------------------------ helpers
    @property
    def bounds(self):
        return self.wind.extent

    def in_bounds(self, x, y):
        xmin, xmax, ymin, ymax = self.wind.extent
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)

    def dist_to_goal(self, state=None):
        s = self.state if state is None else state
        return float(np.hypot(s[0] - self.goal[0], s[1] - self.goal[1]))

    def _obs(self):
        return np.concatenate((self.state, self.goal)).astype(np.float32)

    def _info(self, **extra):
        info = {
            "J": self.J,
            "t": self.steps * self.p.dt,
            "dist": self.dist_to_goal(),
            "success": False,
            "oob": False,
        }
        info.update(extra)
        return info

    # ---------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        if "wind" in options:
            self.wind = options["wind"]
        elif self.wind_sampler is not None:
            self.wind = self.wind_sampler(self.np_random)

        lo, hi = self.spawn_box
        if "start" in options:
            start = np.asarray(options["start"], dtype=np.float64)
        else:
            start = self.np_random.uniform(lo, hi, size=2)

        if "goal" in options:
            goal = np.asarray(options["goal"], dtype=np.float64)
        else:
            goal = self.np_random.uniform(lo, hi, size=2)
            for _ in range(100):
                if np.linalg.norm(goal - start) >= self.min_start_goal_dist:
                    break
                goal = self.np_random.uniform(lo, hi, size=2)

        v0 = np.asarray(options.get("velocity", (0.0, 0.0)), dtype=np.float64)
        self.state = np.array([start[0], start[1], v0[0], v0[1]], dtype=np.float64)
        self.goal = goal
        self.steps = 0
        self.J = 0.0
        return self._obs(), self._info()

    def step(self, action):
        u = np.clip(np.asarray(action, dtype=np.float64), -self.p.u_max, self.p.u_max)
        x, y, vx, vy = self.state
        wx, wy = self.wind(x, y)
        x1, y1, vx1, vy1 = ship_step(x, y, vx, vy, u[0], u[1], float(wx), float(wy), self.p)

        d_prev = self.dist_to_goal()
        self.state = np.array([x1, y1, vx1, vy1], dtype=np.float64)
        d = self.dist_to_goal()

        cost = float(stage_cost(u[0], u[1], self.p))
        self.J += cost
        self.steps += 1
        reward = -cost + self.target_w * (d_prev - d)

        terminated = False
        success = False
        oob = False
        if d <= self.goal_radius:
            terminated = True
            success = True
            reward += self.goal_bonus
        elif not self.in_bounds(x1, y1):
            terminated = True
            oob = True
            reward -= self.oob_penalty
        truncated = (not terminated) and self.steps >= self.max_steps

        info = self._info(success=success, oob=oob, stage_cost=cost, wind=(float(wx), float(wy)))
        return self._obs(), float(reward), terminated, truncated, info


# Backwards-compatible name used by the legacy scripts: CustomEnv(DICT)
CustomEnv = ShipEnv
