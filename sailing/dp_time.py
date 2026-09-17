"""
Time-dependent dynamic programming for sailing: backward induction over forecast time.

State (x, y, heading k, time layer n), t_n = t0 + n * dt. Weather varies with n, so the value
function has one layer per time step and is computed in a single backward sweep -- no
iteration to convergence, one Bellman backup per layer:

    V_n(s, k) = min over the next heading k' of
        same tack :   dt + V_{n+1}(s + v_n(s,k') * dt          * e(k'), k')
        tack      :   dt + V_{n+1}(s + v_n(s,k') * (dt - T_t)  * e(k'), k')
        gybe      :   dt + V_{n+1}(s + v_n(s,k') * (dt - T_g)  * e(k'), k')   only if both legs downwind

    V = 0 inside the goal disc (any n);  V_N = `big` everywhere else (the forecast horizon).

A manoeuvre costs its penalty as lost moving time inside the step, so the time step must be
longer than the tack/gybe penalty -- the ocean-scale regime, where a step is a fraction of an
hour and a tack a few minutes.

The min over k' is not taken over a K x K penalty matrix. The penalty depends only on whether
the side changes and whether the legs are downwind, so for every node the next-heading
values are reduced to six group minima (side x {same, tack, gybe}) and every old heading picks
from those: O(K N) per layer instead of O(K^2 N). That is what makes ocean grids feasible.

Storage: to steer a boat later every layer is needed. Layers are kept on the CPU in float16
(`store="cpu16"`); `store=None` keeps only the running layer, for pure timing.
"""

import time

import numpy as np
import torch

from sailing.polar import wind_geometry


class SailDPTime:
    def __init__(self, seq, polar, p, goal, nx=121, ny=121, n_headings=36, dt_h=None,
                 horizon_h=None, t0_h=0.0, typical_kts=6.0, big=1e4, store="cpu16",
                 device=None, verbose=False):
        self.seq, self.polar, self.p = seq, polar, p
        self.goal = np.asarray(goal, dtype=np.float64)
        self.big = float(big)
        self.store = store
        self.t0 = float(t0_h)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dev = self.device

        xmin, xmax, ymin, ymax = seq.extent
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.nx, self.ny, self.K = int(nx), int(ny), int(n_headings)
        self.dxg = (xmax - xmin) / (nx - 1)
        self.dyg = (ymax - ymin) / (ny - 1)
        cell_nm = min(self.dxg, self.dyg) * p.nm_per_unit
        # default: about one cell per step at a typical boat speed (CFL ~ 1)
        self.dt = float(dt_h) if dt_h else cell_nm / typical_kts
        if self.dt <= max(p.tack_time, p.gybe_time):
            raise ValueError(f"dt ({self.dt:.3f} h) must exceed the manoeuvre penalty "
                             f"({max(p.tack_time, p.gybe_time)} h); coarsen the grid or pass dt_h")
        horizon = float(horizon_h) if horizon_h else seq.horizon_h - self.t0
        self.N_layers = int(np.floor(horizon / self.dt))
        self.cell_nm = cell_nm

        xs = torch.linspace(xmin, xmax, nx, device=dev)
        ys = torch.linspace(ymin, ymax, ny, device=dev)
        self.xs, self.ys = xs, ys
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        self.X, self.Y = X.reshape(-1), Y.reshape(-1)
        self.N = self.X.numel()
        self.heads = torch.linspace(-np.pi, np.pi, self.K + 1, device=dev)[:-1]
        self.cos_h, self.sin_h = torch.cos(self.heads)[:, None], torch.sin(self.heads)[:, None]
        self.goal_mask = ((self.X - self.goal[0]) ** 2 + (self.Y - self.goal[1]) ** 2) <= p.goal_radius ** 2
        if not bool(self.goal_mask.any()):
            raise ValueError("goal disc contains no grid node; refine the grid or enlarge goal_radius")

        # wind of every forecast slice at every DP node, once
        wxs, wys = [], []
        for f in seq.fields:
            s = f.torch_sampler(dev)
            wx, wy = s(self.X, self.Y)
            wxs.append(wx)
            wys.append(wy)
        self.WXn = torch.stack(wxs)
        self.WYn = torch.stack(wys)
        self.layers = None
        self.solve_s = 0.0
        if verbose:
            print(f"[SailDPTime] {nx}x{ny}x{self.K} x {self.N_layers} layers "
                  f"= {self.K * self.N * self.N_layers / 1e6:,.0f} M state-times | cell {cell_nm:.2f} nm | "
                  f"dt {self.dt:.3f} h | horizon {self.N_layers * self.dt:.0f} h | device {dev}")

    # ----------------------------------------------------------------- helpers
    def _wind_nodes(self, t):
        i, a = self.seq.bracket(t)
        if len(self.seq.t) == 1:
            return self.WXn[0], self.WYn[0]
        return (1 - a) * self.WXn[i] + a * self.WXn[i + 1], (1 - a) * self.WYn[i] + a * self.WYn[i + 1]

    def _locate(self, x, y):
        fx = (x - self.xmin) / self.dxg
        fy = (y - self.ymin) / self.dyg
        i = torch.clamp(torch.floor(fx), 0, self.nx - 2).long()
        j = torch.clamp(torch.floor(fy), 0, self.ny - 2).long()
        tx = torch.clamp(fx - i, 0.0, 1.0)
        ty = torch.clamp(fy - j, 0.0, 1.0)
        base = i * self.ny + j
        oob = (x < self.xmin) | (x > self.xmax) | (y < self.ymin) | (y > self.ymax)
        return base, tx, ty, oob

    def _interp(self, V, base, tx, ty):
        """Bilinear read of V (K, N) at per-heading positions base/tx/ty of shape (K, M)."""
        ny = self.ny
        g = lambda off: torch.gather(V, 1, base + off)
        return ((1 - tx) * (1 - ty) * g(0) + tx * (1 - ty) * g(ny)
                + (1 - tx) * ty * g(1) + tx * ty * g(ny + 1))

    def _move_value(self, Vnext, v, tmove, x=None, y=None):
        """
        Value of moving along each heading for `tmove` hours within a step of `dt` hours
        (the rest, dt - tmove, is manoeuvre time). If the straight move enters the goal disc,
        the value is the exact entry time instead of an interpolated V: the disc spans only a
        couple of cells, and interpolating across its edge reads V > 0 inside it -- noise that
        was large enough to make the greedy policy tack back and forth off the finish.
        """
        x = self.X[None, :] if x is None else x
        y = self.Y[None, :] if y is None else y
        vx, vy = v * self.cos_h, v * self.sin_h
        x1 = x + vx * tmove
        y1 = y + vy * tmove
        base, tx, ty, oob = self._locate(x1, y1)
        val = torch.where(oob, torch.full_like(x1, self.big), self.dt + self._interp(Vnext, base, tx, ty))

        gx, gy, R = float(self.goal[0]), float(self.goal[1]), self.p.goal_radius
        px, py = x - gx, y - gy
        a = vx * vx + vy * vy
        b = 2.0 * (px * vx + py * vy)
        c = px * px + py * py - R * R
        disc = b * b - 4.0 * a * c
        tau = (-b - torch.sqrt(torch.clamp(disc, min=0.0))) / (2.0 * torch.clamp(a, min=1e-15))
        inside = c <= 0.0
        hit = inside | ((disc >= 0.0) & (a > 1e-15) & (tau >= 0.0) & (tau <= tmove))
        tau = torch.where(inside, torch.zeros_like(tau), torch.clamp(tau, min=0.0))
        return torch.where(hit, (self.dt - tmove) + tau, val)

    # ------------------------------------------------------------------- solve
    @torch.no_grad()
    def solve(self, verbose=False, log_every=100):
        p, dev, big = self.p, self.device, self.big
        K, N = self.K, self.N
        t_start = time.perf_counter()
        Vnext = torch.full((K, N), big, device=dev)
        Vnext[:, self.goal_mask] = 0.0
        if self.store == "cpu16":
            self.layers = [None] * (self.N_layers + 1)
            self.layers[self.N_layers] = Vnext.half().cpu()
        inf = torch.tensor(big, device=dev)
        for n in range(self.N_layers - 1, -1, -1):
            t = self.t0 + n * self.dt
            wx, wy = self._wind_nodes(t)
            tws, twa, rel = wind_geometry(wx[None, :], wy[None, :], self.heads[:, None], p.kts_per_wind_unit)
            kts = self.polar.speed_torch(twa, tws)
            v = torch.where(kts >= p.min_speed_kts, kts / p.nm_per_unit, torch.zeros_like(kts))
            side = rel >= 0
            dw = rel.abs() > (np.pi / 2)

            M0 = self._move_value(Vnext, v, self.dt)
            Mt = self._move_value(Vnext, v, self.dt - p.tack_time)
            Mg = self._move_value(Vnext, v, self.dt - p.gybe_time)

            A_p = torch.where(side, M0, inf).min(0).values
            A_n = torch.where(~side, M0, inf).min(0).values
            T_p = torch.where(side, Mt, inf).min(0).values
            T_n = torch.where(~side, Mt, inf).min(0).values
            G_p = torch.where(side & dw, Mg, inf).min(0).values
            G_n = torch.where(~side & dw, Mg, inf).min(0).values
            del M0, Mt, Mg

            same = torch.where(side, A_p[None, :], A_n[None, :])
            tack = torch.where(side, T_n[None, :], T_p[None, :])
            gybe = torch.where(dw, torch.where(side, G_n[None, :], G_p[None, :]), inf)
            V = torch.minimum(torch.minimum(same, tack), gybe).clamp_(max=big)
            V[:, self.goal_mask] = 0.0
            Vnext = V
            if self.store == "cpu16":
                self.layers[n] = V.half().cpu()
            if verbose and (self.N_layers - n) % log_every == 0:
                print(f"  layer {n:5d}/{self.N_layers}  {time.perf_counter() - t_start:6.1f}s", flush=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        self.V0 = Vnext
        self.solve_s = time.perf_counter() - t_start
        return dict(solve_s=self.solve_s, layers=self.N_layers, dt=self.dt, cell_nm=self.cell_nm,
                    state_times=K * N * self.N_layers,
                    stored_gb=(K * N * (self.N_layers + 1) * 2 / 1e9) if self.store == "cpu16" else 0.0)

    # ------------------------------------------------------------------ policy
    def _layer_gpu(self, n):
        n = int(np.clip(n, 0, self.N_layers))
        if getattr(self, "_cached_n", None) != n:
            self._cached = self.layers[n].to(self.device).float()
            self._cached_n = n
        return self._cached

    @torch.no_grad()
    def q_values(self, x, y, t, heading=None):
        """Cost-to-go of each next heading at a continuous state and time."""
        if self.layers is None:
            raise RuntimeError("solve with store='cpu16' to steer a boat")
        p, dev = self.p, self.device
        n = int(np.floor((t - self.t0) / self.dt + 1e-9))
        Vnext = self._layer_gpu(n + 1)
        f = self.seq.at(t)
        wx, wy = f(x, y)
        wx = torch.tensor(float(wx), device=dev)
        wy = torch.tensor(float(wy), device=dev)
        tws, twa, rel_new = wind_geometry(wx, wy, self.heads, p.kts_per_wind_unit)
        kts = self.polar.speed_torch(twa, tws)
        v = torch.where(kts >= p.min_speed_kts, kts / p.nm_per_unit, torch.zeros_like(kts))[:, None]
        xt = torch.full((self.K, 1), float(x), device=dev)
        yt = torch.full((self.K, 1), float(y), device=dev)
        q = self._move_value(Vnext, v, self.dt, xt, yt)[:, 0]
        if heading is not None:
            _, _, rel_old = wind_geometry(wx, wy, torch.tensor(float(heading), device=dev), p.kts_per_wind_unit)
            flip = (rel_old * rel_new) < 0
            both_dw = (rel_old.abs() > np.pi / 2) & (rel_new.abs() > np.pi / 2)
            qt = self._move_value(Vnext, v, self.dt - p.tack_time, xt, yt)[:, 0]
            qg = self._move_value(Vnext, v, self.dt - p.gybe_time, xt, yt)[:, 0]
            q = torch.where(flip, torch.where(both_dw, qg, qt), q)
        return q

    def act(self, x, y, t, heading=None):
        return float(self.heads[int(torch.argmin(self.q_values(x, y, t, heading)))])

    def predicted_time(self, x, y):
        return float(self.q_values(x, y, self.t0, None).min())

    def policy(self):
        def pol(env):
            x, y, h = env.state
            return self.act(float(x), float(y), self.t0 + env.t, None if env.steps == 0 else float(h))
        return pol
