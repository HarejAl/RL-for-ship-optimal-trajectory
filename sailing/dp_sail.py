"""
Dynamic programming for sailing: minimum-time value iteration on (x, y, heading).

Why heading is part of the state: the cost of the next leg depends on the heading the boat
is currently on -- switching side costs a tack or gybe. Without it the optimum chatters.
Unlike the motor-ship DP there is no velocity in the state: a polar boat has no inertia, so
the grid is (nx * ny * K) rather than (nx * ny * nv * nv). That makes this DP much cheaper
than the ship one, which matters for any "learned policy vs DP" speed comparison.

Bellman equation (fixed-distance steps, semi-Lagrangian):

    V(s, k) = min_k'  [ P_s(k -> k')  +  ds / v_s(k')  +  V(s + ds * e(k'), k') ]

    V = 0 inside the goal disc, `big` outside the domain or where the boat cannot move
    P_s   tack / gybe time, decided by the wind at s
    v_s   polar speed along heading k' in the wind at s

Moving a fixed distance `ds` (about one grid cell) rather than for a fixed time keeps the
semi-Lagrangian update consistent: a slow heading costs more time, not a shorter hop that
lands in the same cell. V off-grid is bilinear in (x, y); heading is discrete, so the next
value is read from layer k'.

Only static wind here. Time-varying weather needs time as a further state dimension and a
backward sweep -- the isochrone router handles that in one forward pass instead.
"""

import time

import numpy as np
import torch

from sailing.polar import wind_geometry, maneuver_kind


class SailDP:
    def __init__(self, wind, polar, p, goal, nx=81, ny=81, n_headings=36, step_cells=1.2,
                 big=1e3, device=None, verbose=False):
        self.wind, self.polar, self.p = wind, polar, p
        self.goal = np.asarray(goal, dtype=np.float64)
        self.big = float(big)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dev = self.device
        t0 = time.perf_counter()

        xmin, xmax, ymin, ymax = wind.extent
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.nx, self.ny, self.K = int(nx), int(ny), int(n_headings)
        self.dxg = (xmax - xmin) / (nx - 1)
        self.dyg = (ymax - ymin) / (ny - 1)
        self.ds = step_cells * min(self.dxg, self.dyg)
        self.xs = torch.linspace(xmin, xmax, nx, device=dev)
        self.ys = torch.linspace(ymin, ymax, ny, device=dev)
        X, Y = torch.meshgrid(self.xs, self.ys, indexing="ij")
        X, Y = X.reshape(-1), Y.reshape(-1)
        self.N = X.numel()
        self.heads = torch.linspace(-np.pi, np.pi, self.K + 1, device=dev)[:-1]

        self.sampler = wind.torch_sampler(dev)
        wx, wy = self.sampler(X, Y)
        H = self.heads[:, None]
        tws, twa, rel = wind_geometry(wx[None, :], wy[None, :], H, p.kts_per_wind_unit)   # (K, N)
        kts = polar.speed_torch(twa, tws)
        v = kts / p.nm_per_unit
        self.move_cost = torch.where(kts >= p.min_speed_kts, self.ds / torch.clamp(v, min=1e-9),
                                     torch.full_like(v, self.big))                          # (K, N)
        x1 = X[None, :] + self.ds * torch.cos(H)
        y1 = Y[None, :] + self.ds * torch.sin(H)
        self.idx, self.w, self.oob = self._locate(x1, y1)                                  # (K,N,4)

        kind = maneuver_kind(rel[:, None, :], rel[None, :, :])                              # (K, K, N)
        pen = torch.tensor([0.0, p.tack_time, p.gybe_time], device=dev)
        self.P = pen[kind]                                                                   # (K, K, N)

        self.goal_mask = ((X - self.goal[0]) ** 2 + (Y - self.goal[1]) ** 2) <= p.goal_radius ** 2
        if not bool(self.goal_mask.any()):
            raise ValueError("goal disc contains no grid node; refine the grid or enlarge goal_radius")
        self.V = torch.zeros(self.K, self.N, device=dev)
        self.precompute_s = time.perf_counter() - t0
        self.solve_s = 0.0
        self.iterations = 0
        if verbose:
            print(f"[SailDP] {nx}x{ny}x{self.K} = {self.K * self.N:,} states, ds={self.ds:.3f}, "
                  f"device={dev}, precompute {self.precompute_s:.2f}s")

    def _locate(self, x, y):
        """Flat bilinear corner indices (…,4), weights (…,4), and out-of-domain flag."""
        fx = (x - self.xmin) / self.dxg
        fy = (y - self.ymin) / self.dyg
        i = torch.clamp(torch.floor(fx), 0, self.nx - 2).long()
        j = torch.clamp(torch.floor(fy), 0, self.ny - 2).long()
        tx = torch.clamp(fx - i, 0.0, 1.0)
        ty = torch.clamp(fy - j, 0.0, 1.0)
        base = i * self.ny + j
        idx = torch.stack((base, base + self.ny, base + 1, base + self.ny + 1), dim=-1)
        w = torch.stack(((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty), dim=-1)
        oob = (x < self.xmin) | (x > self.xmax) | (y < self.ymin) | (y > self.ymax)
        return idx, w, oob

    @torch.no_grad()
    def solve(self, max_iter=5000, tol=1e-5, verbose=False):
        t0 = time.perf_counter()
        V = self.V
        K, N = self.K, self.N
        delta, it = float("inf"), 0
        for it in range(1, max_iter + 1):
            nxt = torch.gather(V, 1, self.idx.reshape(K, N * 4)).reshape(K, N, 4)
            M = self.move_cost + (nxt * self.w).sum(-1)
            M = torch.where(self.oob, torch.full_like(M, self.big), M)
            V_new = torch.clamp((self.P + M[None, :, :]).min(dim=1).values, max=self.big)
            V_new[:, self.goal_mask] = 0.0
            delta = (V_new - V).abs().max().item()
            V = V_new
            if delta < tol:
                break
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.V = V
        self.iterations += it
        self.solve_s += time.perf_counter() - t0
        if verbose:
            print(f"[SailDP] {it} iterations, {self.solve_s:.2f}s, max|dV|={delta:.2e}")
        return dict(iterations=it, solve_s=self.solve_s, precompute_s=self.precompute_s,
                    converged=delta < tol, states=K * N)

    # ------------------------------------------------------------------ policy
    @torch.no_grad()
    def q_values(self, x, y, heading=None):
        """Cost-to-go of every next heading from a continuous state. heading=None: no penalty."""
        dev = self.device
        wx, wy = self.wind(x, y)
        wx = torch.tensor(float(wx), device=dev)
        wy = torch.tensor(float(wy), device=dev)
        tws, twa, rel_new = wind_geometry(wx, wy, self.heads, self.p.kts_per_wind_unit)
        kts = self.polar.speed_torch(twa, tws)
        v = kts / self.p.nm_per_unit
        move = torch.where(kts >= self.p.min_speed_kts, self.ds / torch.clamp(v, min=1e-9),
                           torch.full_like(v, self.big))
        x1 = x + self.ds * torch.cos(self.heads)
        y1 = y + self.ds * torch.sin(self.heads)
        idx, w, oob = self._locate(x1, y1)                                    # (K, 4)
        nxt = (self.V.gather(1, idx) * w).sum(-1)                              # layer k' for heading k'
        q = torch.where(oob, torch.full_like(move, self.big), move + nxt)
        if heading is not None:
            _, _, rel_old = wind_geometry(wx, wy, torch.tensor(float(heading), device=dev),
                                          self.p.kts_per_wind_unit)
            kind = maneuver_kind(rel_old.expand_as(rel_new), rel_new)
            pen = torch.tensor([0.0, self.p.tack_time, self.p.gybe_time], device=dev)
            q = q + pen[kind]
        return q

    def act(self, x, y, heading=None):
        return float(self.heads[int(torch.argmin(self.q_values(x, y, heading)))])

    def predicted_time(self, x, y):
        return float(self.q_values(x, y, None).min())

    def value_map(self):
        """min over heading of V on the (x, y) grid, shape (nx, ny), with `big` masked to NaN."""
        v = self.V.min(dim=0).values.reshape(self.nx, self.ny).cpu().numpy()
        return np.where(v >= 0.95 * self.big, np.nan, v)

    def policy(self):
        """A `SailEnv` policy: greedy on V, no manoeuvre penalty on the very first decision."""
        def pol(env):
            x, y, h = env.state
            return self.act(float(x), float(y), None if env.steps == 0 else float(h))
        return pol
