"""
Batched ship environment on the GPU, for massively parallel pure RL (PLAN.md step B1).

Thousands of ships are stepped as one tensor operation. Each ship has its own wind field
(drawn from a `FieldBank` of generated fields that lives on the device), its own windage
ratio kappa, its own wind strength, start and goal. Dynamics, cost and observations are
exactly those of `ShipEnv` + `WindObsWrapper(add_kappa=True)` (checked in
tests/test_gpu_env.py), so a policy trained here is evaluated with the CPU tools and DP.

Reward (exact DP cost plus potential-based shaping, which leaves the optimal policy
unchanged; Ng, Harada & Russell 1999):

    r = -stage_cost + gamma * Phi(s') - Phi(s),     Phi(s) = -time_w * dist(s, goal) / V*

plus `oob_penalty` when the ship leaves the map. Arrival is a termination with no bonus:
the agent is paid for reaching the goal only by ending its running cost early.

Actions are normalised to [-1, 1] per axis and scaled by u_max inside the environment.
"""

import numpy as np
import torch

from dynamics import ShipParams
from wind import generate_wind_field
from wind_obs import POS_SCALE, VEL_SCALE, WIND_SCALE, KAPPA_SCALE, TRAIN_SEED_BASE


class FieldBank:
    """F generated wind fields stacked on the device. All share the generator's grid."""

    def __init__(self, n_fields, device, seed_start=TRAIN_SEED_BASE, global_res=16, fields=None,
                 gen_kwargs=None):
        self.device = torch.device(device)
        self.gen_kwargs = dict(gen_kwargs or {})
        self.global_res = int(global_res)
        self.next_seed = int(seed_start)
        f0 = fields[0] if fields is not None else generate_wind_field(seed_start, **self.gen_kwargs)
        self.x0, self.y0 = float(f0.x[0]), float(f0.y[0])
        self.dx, self.dy = f0.dx, f0.dy
        self.nx, self.ny = f0.x.size, f0.y.size
        self.extent = f0.extent
        n = len(fields) if fields is not None else int(n_fields)
        self.n = n
        self.wx = torch.zeros(n, self.nx, self.ny, device=self.device)
        self.wy = torch.zeros_like(self.wx)
        g = self.global_res
        xmin, xmax, ymin, ymax = self.extent
        gx = torch.linspace(xmin, xmax, g, dtype=torch.float64)
        gy = torch.linspace(ymin, ymax, g, dtype=torch.float64)
        GX, GY = torch.meshgrid(gx, gy, indexing="ij")
        self.GX = GX.to(self.device, torch.float32)
        self.GY = GY.to(self.device, torch.float32)
        self.gmap = torch.zeros(n, 2, g, g, device=self.device)   # raw wind on the global grid
        self.seeds = np.full(n, -1, dtype=np.int64)
        if fields is not None:
            for i, f in enumerate(fields):
                self._put(i, f)
        else:
            self.replace(np.arange(n))

    def _put(self, i, f):
        if f.wx.shape != (self.nx, self.ny) or abs(f.x[0] - self.x0) > 1e-9 or abs(f.dx - self.dx) > 1e-12:
            raise ValueError("all fields in a bank must share one grid")
        self.wx[i] = torch.as_tensor(f.wx, dtype=torch.float32, device=self.device)
        self.wy[i] = torch.as_tensor(f.wy, dtype=torch.float32, device=self.device)
        GX = self.GX.double().cpu().numpy()
        GY = self.GY.double().cpu().numpy()
        gx, gy = f(GX, GY)     # same float64 bilinear interpolation as WindObsWrapper
        self.gmap[i, 0] = torch.as_tensor(gx, dtype=torch.float32, device=self.device)
        self.gmap[i, 1] = torch.as_tensor(gy, dtype=torch.float32, device=self.device)

    def replace(self, idx):
        """Regenerate fields `idx` with fresh, never-used training seeds."""
        for i in np.atleast_1d(idx):
            self.seeds[i] = self.next_seed
            self._put(int(i), generate_wind_field(self.next_seed, **self.gen_kwargs))
            self.next_seed += 1

    def sample(self, fidx, px, py):
        """Bilinear wind (wx, wy) of field `fidx` at (px, py); fidx broadcasts against px."""
        fx = (px - self.x0) / self.dx
        fy = (py - self.y0) / self.dy
        i = torch.clamp(torch.floor(fx), 0, self.nx - 2)
        j = torch.clamp(torch.floor(fy), 0, self.ny - 2)
        tx = torch.clamp(fx - i, 0.0, 1.0)
        ty = torch.clamp(fy - j, 0.0, 1.0)
        i = i.long()
        j = j.long()
        f = torch.broadcast_to(fidx, px.shape).long()
        base = (f * self.nx + i) * self.ny + j
        out = []
        for comp in (self.wx.reshape(-1), self.wy.reshape(-1)):
            w00 = comp[base]
            w10 = comp[base + self.ny]
            w01 = comp[base + 1]
            w11 = comp[base + self.ny + 1]
            out.append((1 - tx) * (1 - ty) * w00 + tx * (1 - ty) * w10
                       + (1 - tx) * ty * w01 + tx * ty * w11)
        return out[0], out[1]


class BatchShipEnv:
    """N ships stepped in parallel; finished ships are reset automatically."""

    def __init__(self, n_envs, bank, params=None, kappa_range=(0.05, 0.6), wind_mult_range=(0.25, 1.0),
                 spawn_box=(0.0, 10.0), goal_radius=0.5, min_start_goal_dist=4.0, max_steps=600,
                 local_size=2.0, local_res=16, global_res=16, blob_sigma=0.6, gamma=0.995,
                 oob_penalty=20.0, seed=0):
        self.n = int(n_envs)
        self.bank = bank
        self.dev = bank.device
        self.p = params or ShipParams()
        if self.p.scales().speed != ShipParams().scales().speed:
            raise ValueError("train in reference-ship units: vary kappa and the wind, not V*")
        self.kappa_range = tuple(kappa_range)
        self.wind_mult_range = tuple(wind_mult_range)
        self.spawn_box = spawn_box
        self.goal_radius = float(goal_radius)
        self.radius = float(goal_radius)          # current (curriculum) radius
        self.min_dist = float(min_start_goal_dist)
        self.max_steps = int(max_steps)
        self.gamma = float(gamma)
        self.oob_penalty = float(oob_penalty)
        self.v_ref = self.p.scales().speed
        self.blob_s2 = 2.0 * blob_sigma ** 2
        if global_res != bank.global_res:
            raise ValueError("global_res must match the field bank")
        off = torch.linspace(-local_size, local_size, local_res, dtype=torch.float64)
        ox, oy = torch.meshgrid(off, off, indexing="ij")
        self.ox = ox.to(self.dev, torch.float32)
        self.oy = oy.to(self.dev, torch.float32)
        self.cfg = dict(local_size=float(local_size), local_res=int(local_res), global_res=int(global_res),
                        use_global=True, blob_sigma=float(blob_sigma), add_kappa=True)
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(int(seed))

        z = lambda *s: torch.zeros(*s, device=self.dev)
        self.pos, self.vel, self.goal = z(self.n, 2), z(self.n, 2), z(self.n, 2)
        self.fidx = torch.zeros(self.n, dtype=torch.long, device=self.dev)
        self.mult, self.cd_air = z(self.n), z(self.n)
        self.steps = torch.zeros(self.n, dtype=torch.long, device=self.dev)
        self.J = z(self.n)
        self.reset_idx(torch.ones(self.n, dtype=torch.bool, device=self.dev))

    # ------------------------------------------------------------------ resets
    def _u(self, *shape, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(*shape, generator=self.gen, device=self.dev)

    def reset_idx(self, mask):
        idx = mask.nonzero(as_tuple=True)[0]
        k = idx.numel()
        if k == 0:
            return
        lo, hi = self.spawn_box
        cand = 16
        s = self._u(k, cand, 2, lo=lo, hi=hi)
        g = self._u(k, cand, 2, lo=lo, hi=hi)
        ok = (s - g).norm(dim=-1) >= self.min_dist
        first = torch.where(ok.any(1), ok.float().argmax(1), torch.full_like(ok[:, 0], cand - 1, dtype=torch.long))
        ar = torch.arange(k, device=self.dev)
        self.pos[idx] = s[ar, first]
        self.goal[idx] = g[ar, first]
        self.vel[idx] = 0.0
        self.fidx[idx] = torch.randint(0, self.bank.n, (k,), generator=self.gen, device=self.dev)
        self.mult[idx] = self._u(k, lo=self.wind_mult_range[0], hi=self.wind_mult_range[1])
        self.cd_air[idx] = self.p.cd_water * self._u(k, lo=self.kappa_range[0], hi=self.kappa_range[1])
        self.steps[idx] = 0
        self.J[idx] = 0.0

    def set_cases(self, fidx, start, goal, kappa, mult):
        """Pin every ship to a given case (evaluation). All arguments are length-N sequences."""
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.dev)
        self.fidx[:] = t(fidx, torch.long)
        self.pos[:] = t(start)
        self.goal[:] = t(goal)
        self.vel[:] = 0.0
        self.cd_air[:] = self.p.cd_water * t(kappa)
        self.mult[:] = t(mult)
        self.steps[:] = 0
        self.J[:] = 0.0

    # ------------------------------------------------------------ observation
    def wind_at(self, px, py, fidx=None, mult=None):
        fidx = self.fidx if fidx is None else fidx
        mult = self.mult if mult is None else mult
        shape = (-1,) + (1,) * (px.dim() - 1)
        wx, wy = self.bank.sample(fidx.view(shape), px, py)
        m = mult.view(shape)
        return wx * m, wy * m

    def obs(self):
        x, y = self.pos[:, 0], self.pos[:, 1]
        gx, gy = self.goal[:, 0], self.goal[:, 1]
        vec = torch.stack([(gx - x) / POS_SCALE, (gy - y) / POS_SCALE,
                           self.vel[:, 0] / VEL_SCALE, self.vel[:, 1] / VEL_SCALE,
                           x / POS_SCALE, y / POS_SCALE,
                           (self.cd_air / self.p.cd_water) / KAPPA_SCALE], dim=1)
        px = x[:, None, None] + self.ox
        py = y[:, None, None] + self.oy
        wx, wy = self.wind_at(px, py)
        xmin, xmax, ymin, ymax = self.bank.extent
        inside = ((px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)).float()
        local = torch.stack([wx / WIND_SCALE, wy / WIND_SCALE, inside], dim=1)
        GX, GY = self.bank.GX, self.bank.GY
        wmap = self.bank.gmap[self.fidx] * (self.mult / WIND_SCALE).view(-1, 1, 1, 1)
        ship = torch.exp(-((GX - x.view(-1, 1, 1)) ** 2 + (GY - y.view(-1, 1, 1)) ** 2) / self.blob_s2)
        goal = torch.exp(-((GX - gx.view(-1, 1, 1)) ** 2 + (GY - gy.view(-1, 1, 1)) ** 2) / self.blob_s2)
        glob = torch.cat([wmap, ship[:, None], goal[:, None]], dim=1)
        return {"vec": vec, "local": local, "global": glob}

    # ------------------------------------------------------------------- step
    def dist(self):
        return (self.goal - self.pos).norm(dim=1)

    def step(self, action, auto_reset=True):
        """action: (N, 2) in [-1, 1]. Returns obs, reward, terminated, truncated, info.
        info['final_obs'] is the observation BEFORE the automatic reset (for bootstrapping
        truncated episodes); info['J'], ['t'], ['success'], ['oob'] describe finished episodes."""
        p = self.p
        u = torch.clamp(action, -1.0, 1.0) * p.u_max
        w = torch.stack(self.wind_at(self.pos[:, 0], self.pos[:, 1]), dim=1)
        v = self.vel
        speed = v.norm(dim=1, keepdim=True)
        rv = v - w
        rs = rv.norm(dim=1, keepdim=True)
        acc = u - p.cd_water * speed * v - self.cd_air[:, None] * rs * rv
        d_prev = self.dist()
        self.vel = v + p.dt * acc
        self.pos = self.pos + p.dt * self.vel
        d = self.dist()
        cost = p.dt * (p.time_w + p.ctrl_w * (u * u).sum(dim=1))
        self.J += cost
        self.steps += 1
        reward = -cost + p.time_w * (d_prev - self.gamma * d) / self.v_ref
        success = d <= self.radius
        xmin, xmax, ymin, ymax = self.bank.extent
        x, y = self.pos[:, 0], self.pos[:, 1]
        oob = ~success & ((x < xmin) | (x > xmax) | (y < ymin) | (y > ymax))
        reward = reward - self.oob_penalty * oob.float()
        terminated = success | oob
        truncated = ~terminated & (self.steps >= self.max_steps)
        done = terminated | truncated
        info = dict(success=success, oob=oob, J=self.J.clone(), t=self.steps.float() * p.dt, done=done)
        if truncated.any():
            info["final_obs"] = self.obs()
        if auto_reset and done.any():
            self.reset_idx(done)
        return self.obs(), reward, terminated, truncated, info
