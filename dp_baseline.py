"""
Dynamic-programming baseline: semi-Lagrangian value iteration on a 4D state grid.

The state is (x, y, vx, vy) and the control is the thrust (ux, uy) taken from a
finite grid. The value function V(s) = minimal cost-to-go to the goal disc is
computed by fixed-point iteration of the Bellman equation

    V(s) = min_a [ l(a) + V(f(s, a)) ],      V = 0 on the goal, V = oob_cost outside

where f is exactly the `ship_step` used by the Gymnasium environment and V is
evaluated off-grid by multilinear interpolation (semi-Lagrangian scheme).
Because the dynamics do not depend on V, the successor cell and interpolation
weights of every (state, action) pair are precomputed once; each iteration is
then 16 gathers and a min over actions, vectorised in torch (GPU if available).

The resulting policy is greedy one-step lookahead on V, evaluated on a (finer)
action grid, and is executed in the environment so that both the baseline and
the RL agent are scored on identical dynamics and cost.

The solve must be redone for every (wind field, goal) pair. Its wall-clock time
is the quantity an amortised RL policy is compared against.
"""

import time

import numpy as np
import torch

from dynamics import ShipParams, ship_step, stage_cost, terminal_speed


class ValueIterationPlanner:
    def __init__(
        self,
        wind,
        goal,
        params=None,
        nx=61,
        ny=61,
        nv=13,
        v_max=None,
        n_act=5,
        exec_n_act=9,
        goal_radius=0.5,
        oob_cost=100.0,
        device=None,
        weight_mode="auto",
        verbose=False,
    ):
        """
        weight_mode : 'fp32' | 'fp16' | 'none' | 'auto'. Precomputing the 16 interpolation
                      weights per (state, action) makes an iteration ~3x faster but costs
                      A*N*16*(4|2) bytes; 'auto' picks fp32, then fp16, then none by memory.
        """
        self.wind = wind
        self.goal = np.asarray(goal, dtype=np.float64)
        self.p = params or ShipParams()
        self.goal_radius = goal_radius
        self.oob_cost = float(oob_cost)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dev = self.device

        xmin, xmax, ymin, ymax = wind.extent
        if v_max is None:
            v_max = 1.1 * terminal_speed(self.p, float(wind.speed.max()))
        self.v_max = float(v_max)
        self.nx, self.ny, self.nv = int(nx), int(ny), int(nv)
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.dxg = (xmax - xmin) / (nx - 1)
        self.dyg = (ymax - ymin) / (ny - 1)
        self.dvg = 2.0 * self.v_max / (nv - 1)
        self.N = self.nx * self.ny * self.nv * self.nv

        self.xs = torch.linspace(xmin, xmax, nx, device=dev)
        self.ys = torch.linspace(ymin, ymax, ny, device=dev)
        self.vs = torch.linspace(-self.v_max, self.v_max, nv, device=dev)
        X, Y, VX, VY = torch.meshgrid(self.xs, self.ys, self.vs, self.vs, indexing="ij")
        X, Y, VX, VY = (t.reshape(-1) for t in (X, Y, VX, VY))

        self.sampler = wind.torch_sampler(dev)
        wx, wy = self.sampler(X, Y)

        gx, gy = self.goal
        self.goal_mask = ((X - gx) ** 2 + (Y - gy) ** 2) <= goal_radius ** 2
        if not bool(self.goal_mask.any()):
            raise ValueError("goal disc contains no grid node; increase nx/ny or goal_radius")

        self.actions = self._action_grid(n_act)
        self.exec_actions = self._action_grid(exec_n_act)

        # 16 hypercube corners: bit pattern and flat-index offset
        bits = [(bx, by, bk, bl) for bx in (0, 1) for by in (0, 1) for bk in (0, 1) for bl in (0, 1)]
        self.bits = bits
        nv = self.nv
        self.offsets = torch.tensor(
            [bx * self.ny * nv * nv + by * nv * nv + bk * nv + bl for bx, by, bk, bl in bits],
            device=dev, dtype=torch.long,
        )

        # precompute successor interpolation data for every (state, action)
        t0 = time.perf_counter()
        idx_l, frac_l, oob_l = [], [], []
        for a in range(self.actions.shape[0]):
            ux, uy = self.actions[a, 0], self.actions[a, 1]
            x1, y1, vx1, vy1 = ship_step(X, Y, VX, VY, ux, uy, wx, wy, self.p)
            idx, frac, oob = self._locate(x1, y1, vx1, vy1)
            idx_l.append(idx)
            frac_l.append(frac)
            oob_l.append(oob)
        self.idx = torch.stack(idx_l)          # (A, N) long
        self.frac = torch.stack(frac_l)        # (A, N, 4) float
        self.oob = torch.stack(oob_l)          # (A, N) bool
        self.act_cost = stage_cost(self.actions[:, 0], self.actions[:, 1], self.p)  # (A,)
        self.weight_mode = "none"
        self.weights = self._precompute_weights(weight_mode)  # (16, A, N) or None
        self.precompute_time = time.perf_counter() - t0

        self.V = torch.zeros(self.N, device=dev)
        self.history = []
        self.iterations = 0
        self.solve_time = 0.0
        if verbose:
            print(f"[VI] grid {nx}x{ny}x{nv}x{nv} = {self.N:,} states, "
                  f"{self.actions.shape[0]} actions, v_max={self.v_max:.2f}, "
                  f"device={dev}, weights={getattr(self, 'weight_mode', 'none')}, "
                  f"precompute {self.precompute_time:.2f}s")

    # ----------------------------------------------------------------- grids
    def _action_grid(self, n):
        ua = torch.linspace(-self.p.u_max, self.p.u_max, n, device=self.device)
        UX, UY = torch.meshgrid(ua, ua, indexing="ij")
        return torch.stack((UX.reshape(-1), UY.reshape(-1)), dim=1)

    def _locate(self, x, y, vx, vy):
        """Flat base index, fractional coordinates and out-of-bounds flag of points."""
        fx = (x - self.xmin) / self.dxg
        fy = (y - self.ymin) / self.dyg
        fk = (vx + self.v_max) / self.dvg
        fl = (vy + self.v_max) / self.dvg
        i = torch.clamp(torch.floor(fx), 0, self.nx - 2).long()
        j = torch.clamp(torch.floor(fy), 0, self.ny - 2).long()
        k = torch.clamp(torch.floor(fk), 0, self.nv - 2).long()
        l = torch.clamp(torch.floor(fl), 0, self.nv - 2).long()
        frac = torch.stack((
            torch.clamp(fx - i, 0.0, 1.0),
            torch.clamp(fy - j, 0.0, 1.0),
            torch.clamp(fk - k, 0.0, 1.0),
            torch.clamp(fl - l, 0.0, 1.0),
        ), dim=-1)
        oob = (x < self.xmin) | (x > self.xmax) | (y < self.ymin) | (y > self.ymax)
        idx = ((i * self.ny + j) * self.nv + k) * self.nv + l
        return idx, frac, oob

    def _interp(self, V, idx, frac):
        """Multilinear interpolation of flat V at (idx, frac) of any leading shape."""
        out = torch.zeros_like(frac[..., 0])
        for c, b in enumerate(self.bits):
            w = None
            for d in range(4):
                f = frac[..., d] if b[d] else 1.0 - frac[..., d]
                w = f if w is None else w * f
            out = out + w * V[idx + self.offsets[c]]
        return out

    def _precompute_weights(self, mode, budget_bytes=3e9):
        n = 16 * self.frac.shape[0] * self.frac.shape[1]
        if mode == "auto":
            mode = "fp32" if n * 4 <= budget_bytes else ("fp16" if n * 2 <= budget_bytes else "none")
        if mode == "none":
            return None
        dtype = torch.float32 if mode == "fp32" else torch.float16
        W = torch.empty((16,) + tuple(self.frac.shape[:2]), device=self.device, dtype=dtype)
        for c, b in enumerate(self.bits):
            w = None
            for d in range(4):
                f = self.frac[..., d] if b[d] else 1.0 - self.frac[..., d]
                w = f if w is None else w * f
            W[c] = w.to(dtype)
        self.weight_mode = mode
        return W

    def _interp_all(self, V):
        """Interpolate V at every precomputed successor; returns (A, N)."""
        if self.weights is None:
            return self._interp(V, self.idx, self.frac)
        out = torch.zeros(self.idx.shape, device=self.device, dtype=V.dtype)
        for c in range(16):
            # V[idx + off] == V[off:][idx] since every corner index is inside the grid
            g = V[int(self.offsets[c]):][self.idx]
            w = self.weights[c]
            out.addcmul_(w if w.dtype == V.dtype else w.to(V.dtype), g)
        return out

    # ----------------------------------------------------------------- solve
    @torch.no_grad()
    def solve(self, max_iter=3000, tol=1e-4, verbose=True, log_every=100):
        """Jacobi value iteration until max |V_new - V| < tol. Returns solve statistics."""
        t0 = time.perf_counter()
        V = self.V
        delta = float("inf")
        it = 0
        for it in range(1, max_iter + 1):
            Vn = self._interp_all(V)                                          # (A, N)
            Q = self.act_cost[:, None] + torch.where(self.oob, self.oob_cost, Vn)
            V_new = Q.min(dim=0).values
            V_new[self.goal_mask] = 0.0
            delta = (V_new - V).abs().max().item()
            V = V_new
            self.history.append(delta)
            if verbose and (it % log_every == 0 or it == 1):
                print(f"[VI] iter {it:5d}  max|dV| = {delta:.3e}  "
                      f"elapsed {time.perf_counter() - t0:.1f}s")
            if delta < tol:
                break
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.V = V
        self.iterations += it
        self.solve_time += time.perf_counter() - t0
        stats = dict(iterations=it, time=self.solve_time, converged=delta < tol, delta=delta,
                     precompute_time=self.precompute_time, n_states=self.N,
                     n_actions=int(self.actions.shape[0]))
        if verbose:
            print(f"[VI] done: {it} iterations in {stats['time']:.2f}s, "
                  f"converged={stats['converged']} (max|dV|={delta:.2e})")
        return stats

    # ---------------------------------------------------------------- policy
    @torch.no_grad()
    def q_values(self, state, actions=None):
        """One-step lookahead cost l(a) + V(f(s,a)) for each action. Returns a torch (M,) tensor."""
        actions = self.exec_actions if actions is None else actions
        s = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device)
        wx, wy = self.sampler(s[0:1], s[1:2])
        x1, y1, vx1, vy1 = ship_step(s[0], s[1], s[2], s[3], actions[:, 0], actions[:, 1],
                                     wx, wy, self.p)
        idx, frac, oob = self._locate(x1, y1, vx1, vy1)
        Vn = self._interp(self.V, idx, frac)
        return stage_cost(actions[:, 0], actions[:, 1], self.p) + torch.where(oob, self.oob_cost, Vn)

    def act(self, state):
        q = self.q_values(state)
        return self.exec_actions[int(q.argmin())].cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def q_values_batch(self, states, actions=None):
        """Q for a batch of states (B, 4) and every action (M, 2); returns (B, M)."""
        actions = self.exec_actions if actions is None else actions
        s = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
        wx, wy = self.sampler(s[:, 0], s[:, 1])
        x, y, vx, vy = (s[:, i:i + 1] for i in range(4))
        ux, uy = actions[None, :, 0], actions[None, :, 1]
        x1, y1, vx1, vy1 = ship_step(x, y, vx, vy, ux, uy, wx[:, None], wy[:, None], self.p)
        idx, frac, oob = self._locate(x1, y1, vx1, vy1)
        Vn = self._interp(self.V, idx, frac)
        return stage_cost(ux, uy, self.p) + torch.where(oob, self.oob_cost, Vn)

    def act_batch(self, states):
        """Greedy actions (B, 2) and their Q values (B,) for a batch of states."""
        q = self.q_values_batch(states)
        best = q.argmin(dim=1)
        return (self.exec_actions[best].cpu().numpy().astype(np.float64),
                q.gather(1, best[:, None])[:, 0].cpu().numpy())

    @torch.no_grad()
    def value(self, x, y, vx=0.0, vy=0.0):
        """Interpolated value at arbitrary (broadcastable) numpy coordinates."""
        x, y, vx, vy = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float),
                                           np.asarray(vx, float), np.asarray(vy, float))
        to = lambda a: torch.as_tensor(a.astype(np.float32).reshape(-1), device=self.device)
        idx, frac, oob = self._locate(to(x), to(y), to(vx), to(vy))
        v = torch.where(oob, self.oob_cost, self._interp(self.V, idx, frac))
        return v.cpu().numpy().reshape(x.shape)

    def value_slice(self, vx=0.0, vy=0.0):
        """V on the (x, y) grid at the velocity node nearest to (vx, vy); shape (nx, ny)."""
        k = int(np.argmin(np.abs(self.vs.cpu().numpy() - vx)))
        l = int(np.argmin(np.abs(self.vs.cpu().numpy() - vy)))
        return self.V.reshape(self.nx, self.ny, self.nv, self.nv)[:, :, k, l].cpu().numpy()

    # --------------------------------------------------------------- rollout
    def rollout(self, env, start, velocity=(0.0, 0.0), max_steps=None):
        """Execute the greedy policy in `env` from `start`; returns trajectory and cost."""
        max_steps = max_steps or env.max_steps
        obs, info = env.reset(options=dict(start=start, goal=self.goal, wind=self.wind,
                                           velocity=velocity))
        traj = [env.state.copy()]
        actions, rewards = [], []
        terminated = truncated = False
        for _ in range(max_steps):
            u = self.act(env.state)
            obs, r, terminated, truncated, info = env.step(u)
            traj.append(env.state.copy())
            actions.append(u)
            rewards.append(r)
            if terminated or truncated:
                break
        return dict(
            J=info["J"], t=info["t"], success=info["success"], oob=info["oob"],
            steps=len(actions), return_=float(np.sum(rewards)), terminated=bool(terminated),
            traj=np.array(traj), actions=np.array(actions), rewards=np.array(rewards),
        )
