"""
Time-varying wind for sailing: a sequence of forecast slices on one grid, interpolated in time.

    seq = WindSequence.from_openmeteo((48, 56), (-25, -12), days=7)    # real, cached on disk
    seq = WindSequence.synthetic(seed=0, horizon_h=120)                 # evolving GRF, no API
    p   = sail_params_for(seq)                                          # physical scales from meta
    field = seq.at(37.5)                                                # WindField at t = 37.5 h

Real forecasts carry `km_per_unit` and `ms_per_unit` in their metadata; `sail_params_for`
turns those into `nm_per_unit` and `kts_per_wind_unit`, so a route through a real forecast is
measured in real nautical miles, knots and hours.
"""

import os

import numpy as np

from wind import WindField, generate_wind_field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "..", "output", "cache")
KNOT_MS = 0.514444
KM_PER_NM = 1.852


class WindSequence:
    """Forecast slices (t_hours, WindField) sharing one grid; linear in time, clamped at the ends."""

    def __init__(self, times_h, fields, labels=None, meta=None):
        self.t = np.asarray(times_h, dtype=np.float64)
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("slice times must be strictly increasing")
        self.fields = list(fields)
        f0 = self.fields[0]
        self.x, self.y = f0.x, f0.y
        self.WX = np.stack([f.wx for f in self.fields])
        self.WY = np.stack([f.wy for f in self.fields])
        self.labels = labels or [f"+{t:g} h" for t in self.t]
        self.meta = dict(meta if meta is not None else f0.meta)
        self._cache = {}

    @property
    def extent(self):
        return self.fields[0].extent

    @property
    def horizon_h(self):
        return float(self.t[-1])

    def bracket(self, t):
        """(i, a): the field at t is (1-a) * slice i + a * slice i+1."""
        if len(self.t) == 1:
            return 0, 0.0
        tc = float(np.clip(t, self.t[0], self.t[-1]))
        i = int(np.clip(np.searchsorted(self.t, tc, side="right") - 1, 0, len(self.t) - 2))
        a = (tc - self.t[i]) / (self.t[i + 1] - self.t[i])
        return i, float(np.clip(a, 0.0, 1.0))

    def at(self, t):
        key = round(float(t), 4)
        if key in self._cache:
            return self._cache[key]
        i, a = self.bracket(t)
        if len(self.t) == 1:
            wf = self.fields[0]
        else:
            wf = WindField(self.x, self.y, (1 - a) * self.WX[i] + a * self.WX[i + 1],
                           (1 - a) * self.WY[i] + a * self.WY[i + 1], meta=self.meta)
        if len(self._cache) > 5000:
            self._cache.clear()
        self._cache[key] = wf
        return wf

    def __call__(self, t):
        return self.at(t)

    # ------------------------------------------------------------ constructors
    @classmethod
    def constant(cls, field, horizon_h):
        return cls([0.0, float(horizon_h)], [field, field], meta=field.meta)

    @classmethod
    def synthetic(cls, seed=0, horizon_h=120.0, slice_h=12.0, nx=41, ny=41, drift=0.06):
        """
        Evolving weather without an API: a random field every `slice_h` hours, blended in time,
        plus a large-scale pattern that drifts east by `drift` domain units per hour so the
        flow genuinely moves rather than just flickering.
        """
        n = int(np.ceil(horizon_h / slice_h)) + 1
        times, fields = [], []
        base = generate_wind_field(seed, nx=nx, ny=ny)
        for i in range(n):
            t = i * slice_h
            f = generate_wind_field(seed * 1000 + i, nx=nx, ny=ny)
            shift = drift * t
            wx = 0.55 * f.wx + 0.45 * _shifted(base.wx, base.x, shift)
            wy = 0.55 * f.wy + 0.45 * _shifted(base.wy, base.x, shift)
            times.append(t)
            fields.append(WindField(base.x, base.y, wx, wy))
        meta = dict(source="synthetic", km_per_unit=18.52, ms_per_unit=KNOT_MS * 2.0)
        return cls(times, fields, meta=meta)

    @classmethod
    def from_cache_npz(cls, path):
        with np.load(path, allow_pickle=True) as f:
            x, y, WX, WY = f["x"], f["y"], f["wx"], f["wy"]
            labels = [str(t) for t in f["times"]]
            meta = dict(f["meta"].item())
        fields = [WindField(x, y, WX[i], WY[i], meta=meta) for i in range(len(labels))]
        return cls(np.arange(len(labels), dtype=float), fields, labels=labels, meta=meta)

    @classmethod
    def from_openmeteo(cls, lat_range, lon_range, days=7, nx=20, ny=20, ref_speed=25.0, name=None):
        """Hourly real forecast, fetched once and cached under output/cache/."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        name = name or f"sail_{lat_range[0]:g}_{lat_range[1]:g}_{lon_range[0]:g}_{lon_range[1]:g}"
        path = os.path.join(CACHE_DIR, f"{name}_{nx}x{ny}_d{days}_ref{ref_speed:g}.npz")
        if not os.path.exists(path):
            slices = WindField.from_openmeteo_sequence(
                lat_range, lon_range, nx=nx, ny=ny, hours=list(range(24 * days)),
                forecast_days=days, ref_speed=ref_speed)
            f0 = slices[0][1]
            np.savez_compressed(path, x=f0.x, y=f0.y, wx=np.stack([f.wx for _, f in slices]),
                                wy=np.stack([f.wy for _, f in slices]),
                                times=np.array([t for t, _ in slices], dtype=object),
                                meta=np.array(f0.meta, dtype=object))
        return cls.from_cache_npz(path)


def _shifted(arr, x, shift):
    """Periodically shift a field east by `shift` domain units (row-wise interpolation)."""
    span = x[-1] - x[0]
    xs = (x - x[0] - shift) % span + x[0]
    return np.stack([np.interp(xs, x, arr[:, j]) for j in range(arr.shape[1])], axis=1)


def sail_params_for(seq_or_meta, **overrides):
    """SailParams whose distance and wind scales come from real-forecast metadata."""
    from sailing.boat_env import SailParams
    meta = seq_or_meta.meta if hasattr(seq_or_meta, "meta") else seq_or_meta
    kw = dict(nm_per_unit=meta["km_per_unit"] / KM_PER_NM,
              kts_per_wind_unit=meta["ms_per_unit"] / KNOT_MS)
    kw.update(overrides)
    return SailParams(**kw)
