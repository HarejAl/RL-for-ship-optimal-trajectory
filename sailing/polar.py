"""
Polar performance model of a sailing yacht, plus the wind geometry shared by every planner.

A polar gives boat speed through the water (knots) as a function of
    TWS  true wind speed (knots)
    TWA  true wind angle (degrees, 0 = sailing straight into the wind, 180 = dead downwind)
and is symmetric in port/starboard.

`Polar.synthetic` builds a plausible ~35 ft cruiser: a no-go zone below ~40 deg, fastest on
a beam/broad reach, slower dead downwind (so the best downwind VMG is found by gybing at an
angle, not by running straight), and speed that grows with wind and levels off towards hull
speed. `Polar.from_pol` loads a real table (ORC-style: header row of TWS, one row per TWA).

Both are resampled onto a uniform (TWA, TWS) grid so lookups are plain index arithmetic and
work identically in numpy and torch -- the DP planner evaluates the polar on the GPU.
"""

import numpy as np

TWS_GRID = np.arange(0.0, 40.0 + 1e-9, 0.5)   # knots
TWA_GRID = np.arange(0.0, 180.0 + 1e-9, 1.0)  # degrees


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


class Polar:
    """Boat speed table on a uniform grid: `table[i_twa, j_tws]` in knots."""

    def __init__(self, twa_deg, tws_kts, speed_kts, name="polar"):
        twa = np.asarray(twa_deg, dtype=np.float64)
        tws = np.asarray(tws_kts, dtype=np.float64)
        spd = np.asarray(speed_kts, dtype=np.float64)
        if spd.shape != (twa.size, tws.size):
            raise ValueError("speed table must have shape (len(twa), len(tws))")
        self.name = name
        self.table = self._resample(twa, tws, spd)
        self.d_twa = float(TWA_GRID[1] - TWA_GRID[0])
        self.d_tws = float(TWS_GRID[1] - TWS_GRID[0])
        self._torch = {}

    @staticmethod
    def _resample(twa, tws, spd):
        """Onto TWA_GRID x TWS_GRID. Angles below the first row -> 0 (no data = cannot sail);
        winds above the last column hold the last column; 0 kt wind -> 0 kt boat speed."""
        if tws[0] > 0.0:
            tws = np.concatenate(([0.0], tws))
            spd = np.concatenate((np.zeros((spd.shape[0], 1)), spd), axis=1)
        if twa[0] > 0.0:
            twa = np.concatenate(([0.0], twa))
            spd = np.concatenate((np.zeros((1, spd.shape[1])), spd), axis=0)
        by_tws = np.stack([np.interp(TWS_GRID, tws, row) for row in spd])          # (n_twa_raw, n_tws)
        return np.stack([np.interp(TWA_GRID, twa, by_tws[:, j])
                         for j in range(TWS_GRID.size)], axis=1)                      # (n_twa, n_tws)

    # ------------------------------------------------------------ constructors
    @classmethod
    def synthetic(cls, hull_speed=8.0, no_go=40.0, ramp=6.0, close_ratio=0.62, beam_peak=105.0,
                  run_ratio=0.72, run_power=1.6, w_ref=9.0):
        """
        Parametric cruiser polar.
          angle shape:  0 below `no_go`, smooth ramp over `ramp` deg, rising from `close_ratio`
                        to 1 at `beam_peak`, falling to `run_ratio` dead downwind
          wind shape:   hull_speed * (1 - exp(-TWS / w_ref))   (light air ~0.7 x TWS, saturating)
        """
        th = TWA_GRID
        rise = close_ratio + (1.0 - close_ratio) * np.sin(
            0.5 * np.pi * np.clip((th - no_go) / (beam_peak - no_go), 0.0, 1.0))
        fall = 1.0 - (1.0 - run_ratio) * np.clip((th - beam_peak) / (180.0 - beam_peak), 0.0, 1.0) ** run_power
        shape = np.where(th <= beam_peak, rise, fall) * _smoothstep((th - no_go) / ramp)
        wind = hull_speed * (1.0 - np.exp(-TWS_GRID / w_ref))
        return cls(TWA_GRID, TWS_GRID, np.outer(shape, wind), name="synthetic cruiser")

    @classmethod
    def from_pol(cls, path):
        """
        Load a polar table. Accepted layout (tab, semicolon, comma or whitespace separated):
            TWA\\TWS   6    8    10   ...
            40        4.1  4.9  5.4  ...
            52        4.8  5.7  6.2  ...
        """
        import re
        rows = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = [s for s in re.split(r"[\t;,\s]+", line.strip()) if s]
                if parts:
                    rows.append(parts)
        tws = np.array([float(v) for v in rows[0][1:]])
        twa, spd = [], []
        for r in rows[1:]:
            twa.append(float(r[0]))
            vals = [float(v) for v in r[1:1 + tws.size]]
            spd.append(vals + [0.0] * (tws.size - len(vals)))
        order = np.argsort(twa)
        return cls(np.array(twa)[order], tws, np.array(spd)[order],
                   name=str(path).replace("\\", "/").split("/")[-1])

    def save_pol(self, path, tws_cols=(4, 6, 8, 10, 12, 14, 16, 20, 25), twa_rows=None):
        twa_rows = twa_rows if twa_rows is not None else list(range(0, 181, 5))
        with open(path, "w", encoding="utf-8") as f:
            f.write("TWA\\TWS\t" + "\t".join(f"{c:g}" for c in tws_cols) + "\n")
            for a in twa_rows:
                vals = self.speed(np.full(len(tws_cols), float(a)), np.asarray(tws_cols, float))
                f.write(f"{a:g}\t" + "\t".join(f"{v:.3f}" for v in vals) + "\n")

    # ------------------------------------------------------------------ lookup
    def speed(self, twa_deg, tws_kts):
        """Boat speed (knots), bilinear on the table. numpy, any broadcastable shapes."""
        twa = np.abs((np.asarray(twa_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0)
        tws = np.clip(np.asarray(tws_kts, dtype=np.float64), 0.0, TWS_GRID[-1])
        fa = twa / self.d_twa
        fw = tws / self.d_tws
        ia = np.clip(np.floor(fa).astype(np.int64), 0, TWA_GRID.size - 2)
        iw = np.clip(np.floor(fw).astype(np.int64), 0, TWS_GRID.size - 2)
        ta = np.clip(fa - ia, 0.0, 1.0)
        tw = np.clip(fw - iw, 0.0, 1.0)
        T = self.table
        return ((1 - ta) * (1 - tw) * T[ia, iw] + ta * (1 - tw) * T[ia + 1, iw]
                + (1 - ta) * tw * T[ia, iw + 1] + ta * tw * T[ia + 1, iw + 1])

    def speed_torch(self, twa_deg, tws_kts):
        """Same lookup on torch tensors (the table is cached per device/dtype)."""
        import torch
        key = (str(twa_deg.device), twa_deg.dtype)
        if key not in self._torch:
            self._torch[key] = torch.as_tensor(self.table, device=twa_deg.device, dtype=twa_deg.dtype)
        T = self._torch[key]
        twa = torch.abs(torch.remainder(twa_deg + 180.0, 360.0) - 180.0)
        tws = torch.clamp(tws_kts, 0.0, float(TWS_GRID[-1]))
        fa = twa / self.d_twa
        fw = tws / self.d_tws
        ia = torch.clamp(torch.floor(fa), 0, TWA_GRID.size - 2).long()
        iw = torch.clamp(torch.floor(fw), 0, TWS_GRID.size - 2).long()
        ta = torch.clamp(fa - ia, 0.0, 1.0)
        tw = torch.clamp(fw - iw, 0.0, 1.0)
        return ((1 - ta) * (1 - tw) * T[ia, iw] + ta * (1 - tw) * T[ia + 1, iw]
                + (1 - ta) * tw * T[ia, iw + 1] + ta * tw * T[ia + 1, iw + 1])

    def best_vmg(self, tws_kts, upwind=True):
        """(TWA deg, VMG knots) maximising speed towards (upwind) or away from (downwind) the wind."""
        twa = TWA_GRID
        v = self.speed(twa, np.full_like(twa, float(tws_kts)))
        vmg = v * np.cos(np.deg2rad(twa)) * (1.0 if upwind else -1.0)
        i = int(np.argmax(vmg))
        return float(twa[i]), float(vmg[i])


# ------------------------------------------------------------------ geometry
def wrap_angle(a):
    """Wrap radians to [-pi, pi). Works for numpy and torch (including CUDA tensors)."""
    if isinstance(a, np.ndarray) or np.isscalar(a):
        return a - 2.0 * np.pi * np.floor((a + np.pi) / (2.0 * np.pi))
    import torch
    return a - 2.0 * np.pi * torch.floor((a + np.pi) / (2.0 * np.pi))


def wind_geometry(wx, wy, heading, kts_per_wind_unit):
    """
    For a boat on `heading` in wind (wx, wy) [direction the air moves towards]:
        tws   true wind speed in knots
        twa   true wind angle in degrees, 0 = bow into the wind
        rel   signed angle heading - wind_from, radians in (-pi, pi]; its sign is the tack
    numpy or torch, broadcastable.
    """
    lib = np if isinstance(wx, np.ndarray) or np.isscalar(wx) else __import__("torch")
    tws = lib.sqrt(wx * wx + wy * wy) * kts_per_wind_unit
    wind_from = lib.arctan2(-wy, -wx) if lib is np else lib.atan2(-wy, -wx)
    rel = wrap_angle(heading - wind_from)
    twa = lib.abs(rel) * (180.0 / np.pi)
    return tws, twa, rel


def maneuver_kind(rel_old, rel_new):
    """
    0 = same side, 1 = charged as a tack, 2 = charged as a gybe.

    The side changes when `rel` changes sign. A switch is charged as a GYBE only when both the
    old and the new leg are downwind (TWA > 90 deg); if either leg is upwind it is charged as a
    TACK. Classifying instead by which way the shorter turn goes left a loophole: a planner could
    change from close-hauled to close-hauled by "gybing" the long way round (a 270 deg chicken
    gybe) and pay the cheaper gybe price -- the opposite of reality, where that manoeuvre is
    slower than a tack. All planners and the environment share this rule.
    """
    lib = np if isinstance(rel_old, np.ndarray) or np.isscalar(rel_old) else __import__("torch")
    flip = (rel_old * rel_new) < 0
    both_downwind = (lib.abs(rel_old) > np.pi / 2) & (lib.abs(rel_new) > np.pi / 2)
    if lib is np:
        return np.where(flip, np.where(both_downwind, 2, 1), 0)
    return lib.where(flip, lib.where(both_downwind, 2, 1), 0)
