"""
Wind fields on a regular grid: container, interpolation, I/O and a random generator.

Representation
--------------
A wind field is stored as the two velocity components (wx, wy) on a regular grid
with axes x (shape nx) and y (shape ny); wx and wy have shape (nx, ny), x first.
Interpolating the components is the correct thing to do: the legacy code
interpolated speed and direction separately, which breaks where the direction
wraps around 2*pi.

Legacy convention (WF.pkl): 'Intensity' is the speed and 'Direction' D is such
that the wind blows toward -(sin D, cos D). `from_legacy_dict` and
`to_legacy_dict` convert between the two representations.
"""

import os

import numpy as np
from scipy.ndimage import gaussian_filter

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_FIELD_PATH = os.path.join(SCRIPT_DIR, "WF.pkl")


class WindField:
    """Bilinear-interpolated 2D wind field on a uniform grid."""

    def __init__(self, x, y, wx, wy, meta=None):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.wx = np.asarray(wx, dtype=np.float64)
        self.wy = np.asarray(wy, dtype=np.float64)
        if self.wx.shape != (self.x.size, self.y.size) or self.wy.shape != self.wx.shape:
            raise ValueError("wx and wy must have shape (len(x), len(y))")
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
        if not (np.allclose(np.diff(self.x), self.dx) and np.allclose(np.diff(self.y), self.dy)):
            raise ValueError("grid axes must be uniformly spaced")
        self._torch_cache = {}
        # physical-scale metadata (km_per_unit, ms_per_unit, ...) for fields from real data;
        # empty for synthetic/nondimensional fields. Lets you recover real units after a zoom.
        self.meta = dict(meta or {})

    # ------------------------------------------------------------------ props
    @property
    def extent(self):
        """(xmin, xmax, ymin, ymax)"""
        return float(self.x[0]), float(self.x[-1]), float(self.y[0]), float(self.y[-1])

    @property
    def speed(self):
        return np.hypot(self.wx, self.wy)

    @property
    def direction(self):
        """Legacy direction angle D in [0, 2*pi): wind blows toward -(sin D, cos D)."""
        return np.mod(np.arctan2(-self.wx, -self.wy), 2.0 * np.pi)

    # -------------------------------------------------------------- converters
    @classmethod
    def from_legacy_dict(cls, d):
        inten = np.asarray(d["Intensity"], dtype=np.float64)
        direc = np.asarray(d["Direction"], dtype=np.float64)
        return cls(d["x"], d["y"], -inten * np.sin(direc), -inten * np.cos(direc))

    def to_legacy_dict(self):
        return {"x": self.x.copy(), "y": self.y.copy(),
                "Intensity": self.speed, "Direction": self.direction}

    @classmethod
    def load_legacy(cls, path=LEGACY_FIELD_PATH):
        import pickle
        with open(path, "rb") as f:
            return cls.from_legacy_dict(pickle.load(f))

    def save(self, path):
        np.savez(path, x=self.x, y=self.y, wx=self.wx, wy=self.wy)

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(f["x"], f["y"], f["wx"], f["wy"])

    # ----------------------------------------------------- real forecast data
    EARTH_KM_PER_DEG = 111.32

    @classmethod
    def from_openmeteo(cls, lat_range, lon_range, nx=24, ny=24, hour=0,
                       forecast_days=2, domain_size=10.0, margin_frac=0.1,
                       ref_speed=15.0, wind_ref=10.0, max_speed=None,
                       model=None, timeout=60, return_times=False):
        """
        Build a WindField from Open-Meteo 10 m wind (no API key needed), NORMALISED so a
        policy works at any zoom level.

        lat_range, lon_range : (min, max) degrees of the region to cover.
        nx, ny               : sample resolution (nx along longitude -> x, ny along latitude -> y).
        hour                 : index into the hourly forecast to use.

        Normalisation (this is what makes zoom-invariance work):
          * Space: the region's *longer* physical side maps to `domain_size` model units via a
            length scale L (km per unit); the shorter side keeps its true proportion (aspect is
            preserved, not stretched). A `margin_frac*domain_size` border is added on each side.
            Zooming in/out only changes L -- the ship always sees a `domain_size`-unit world.
          * Wind: divide by a FIXED `ref_speed` (m/s) and multiply by `wind_ref` units, so
            `ref_speed` m/s always maps to `wind_ref` units. Absolute severity is preserved and
            two different regions / zoom levels are directly comparable. (Pass `max_speed`
            instead to fall back to per-field peak rescaling.)

        `field.meta` carries `km_per_unit`, `ms_per_unit`, the box size in km, the lat/lon box
        and the forecast time, so you can always convert model units back to physical units.
        `model` pins one Open-Meteo model (e.g. "ecmwf_ifs025", uniform 0.25 deg ~25 km);
        None uses the default best-match blend (higher resolution near coasts).
        """
        import time
        import requests

        x, y, L, wkm, hkm, mlat, spx, spy = cls._geo_axes(lat_range, lon_range, nx, ny,
                                                          domain_size, margin_frac)
        lats = np.linspace(*sorted(lat_range), ny)
        lons = np.linspace(*sorted(lon_range), nx)
        LO, LA = np.meshgrid(lons, lats, indexing="ij")  # (nx, ny), lon first
        speed, direction, times = cls._fetch_openmeteo(
            LA.ravel(), LO.ravel(), forecast_days, model, timeout, requests, time)
        wx, wy = cls._uv(speed[:, hour].reshape(nx, ny), direction[:, hour].reshape(nx, ny))
        wx, wy, ms_per_unit = cls._normalize_wind(wx, wy, max_speed, ref_speed, wind_ref)
        meta = cls._geo_meta(L, ms_per_unit, wkm, hkm, lat_range, lon_range, mlat, spx, spy,
                             times[hour] if times else None, model)
        field = cls(x, y, wx, wy, meta=meta)
        return (field, times) if return_times else field

    @classmethod
    def from_openmeteo_sequence(cls, lat_range, lon_range, nx=24, ny=24, hours=None,
                                forecast_days=2, domain_size=10.0, margin_frac=0.1,
                                ref_speed=15.0, wind_ref=10.0, max_speed=None,
                                model=None, timeout=60):
        """
        Fetch ONE Open-Meteo query and return a list of (timestamp, WindField) for the given
        `hours` (default every 3 h). This is the time-varying input for a receding-horizon run.
        Same normalisation as `from_openmeteo`: the fixed `ref_speed` (or one shared `max_speed`
        factor) keeps every slice on the same scale so the fields are comparable across time.
        """
        import time
        import requests
        x, y, L, wkm, hkm, mlat, spx, spy = cls._geo_axes(lat_range, lon_range, nx, ny,
                                                          domain_size, margin_frac)
        lats = np.linspace(*sorted(lat_range), ny)
        lons = np.linspace(*sorted(lon_range), nx)
        LO, LA = np.meshgrid(lons, lats, indexing="ij")
        speed, direction, times = cls._fetch_openmeteo(
            LA.ravel(), LO.ravel(), forecast_days, model, timeout, requests, time)
        n_hours = speed.shape[1]
        if hours is None:
            hours = list(range(0, n_hours, 3))
        hours = [h for h in hours if h < n_hours]
        # one shared per-sequence peak scale only when using max_speed (ref_speed is already shared)
        shared = None
        if max_speed is not None and ref_speed is None:
            wxa, wya = cls._uv(speed[:, hours].reshape(nx, ny, -1), direction[:, hours].reshape(nx, ny, -1))
            shared = max_speed / max(np.hypot(wxa, wya).max(), 1e-9)
        out = []
        for h in hours:
            wx, wy = cls._uv(speed[:, h].reshape(nx, ny), direction[:, h].reshape(nx, ny))
            if shared is not None:
                wx, wy, ms_per_unit = wx * shared, wy * shared, 1.0 / shared
            else:
                wx, wy, ms_per_unit = cls._normalize_wind(wx, wy, max_speed, ref_speed, wind_ref)
            meta = cls._geo_meta(L, ms_per_unit, wkm, hkm, lat_range, lon_range, mlat, spx, spy,
                                 times[h] if times else None, model)
            out.append((times[h] if times else h, cls(x, y, wx, wy, meta=meta)))
        return out

    @staticmethod
    def _uv(speed, direction_deg):
        """Meteorological (speed, direction 'from') -> (east, north) velocity components."""
        d = np.deg2rad(np.nan_to_num(direction_deg))
        sp = np.nan_to_num(speed)
        return -sp * np.sin(d), -sp * np.cos(d)

    @staticmethod
    def _normalize_wind(wx, wy, max_speed, ref_speed, wind_ref):
        """Return rescaled (wx, wy) and the resulting m/s per model unit. max_speed (per-field
        peak) takes precedence; else ref_speed maps ref_speed m/s -> wind_ref units; else raw."""
        if max_speed is not None:
            peak = max(np.hypot(wx, wy).max(), 1e-9)
            return wx * (max_speed / peak), wy * (max_speed / peak), peak / max_speed
        if ref_speed is not None:
            s = wind_ref / ref_speed
            return wx * s, wy * s, ref_speed / wind_ref
        return wx, wy, 1.0

    @classmethod
    def _geo_axes(cls, lat_range, lon_range, nx, ny, domain_size, margin_frac):
        """Aspect-preserving model axes for a lat/lon box: the longer physical side spans
        `domain_size` units. Returns x, y, L (km/unit), width_km, height_km, mean_lat, span_x, span_y."""
        lat0, lat1 = sorted(lat_range)
        lon0, lon1 = sorted(lon_range)
        mean_lat = 0.5 * (lat0 + lat1)
        width_km = (lon1 - lon0) * cls.EARTH_KM_PER_DEG * np.cos(np.deg2rad(mean_lat))
        height_km = (lat1 - lat0) * cls.EARTH_KM_PER_DEG
        L = max(width_km, height_km) / domain_size  # km per model unit (from the longer side)
        span_x, span_y = width_km / L, height_km / L
        m = margin_frac * domain_size
        x = np.linspace(-m, span_x + m, nx)
        y = np.linspace(-m, span_y + m, ny)
        return x, y, L, width_km, height_km, mean_lat, span_x, span_y

    @staticmethod
    def _geo_meta(L, ms_per_unit, wkm, hkm, lat_range, lon_range, mean_lat, spx, spy, t, model):
        return dict(source="open-meteo", km_per_unit=round(float(L), 3),
                    ms_per_unit=round(float(ms_per_unit), 4),
                    box_km=(round(float(wkm), 1), round(float(hkm), 1)),
                    domain_span=(round(float(spx), 2), round(float(spy), 2)),
                    lat_range=tuple(sorted(lat_range)), lon_range=tuple(sorted(lon_range)),
                    mean_lat=round(float(mean_lat), 3), time=t, model=model or "best_match")

    @staticmethod
    def _fetch_openmeteo(flat_lat, flat_lon, forecast_days, model, timeout, requests, time):
        """Query Open-Meteo for all points; return (speed, direction) of shape (n_points, n_hours) and times."""
        speed = direction = None
        times = None
        chunk = 200  # locations per request (Open-Meteo accepts many; keep call count low)
        n_chunks = (flat_lat.size + chunk - 1) // chunk
        for ci, s in enumerate(range(0, flat_lat.size, chunk)):
            sl = slice(s, s + chunk)
            params = {
                "latitude": ",".join(f"{v:.4f}" for v in flat_lat[sl]),
                "longitude": ",".join(f"{v:.4f}" for v in flat_lon[sl]),
                "hourly": "wind_speed_10m,wind_direction_10m",
                "forecast_days": forecast_days,
                "wind_speed_unit": "ms",
            }
            if model:
                params["models"] = model
            for attempt in range(6):  # exponential backoff on rate limiting / transient errors
                r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                r.raise_for_status()
                break
            else:
                r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict):
                payload = [payload]
            for k, pt in enumerate(payload):
                h = pt["hourly"]
                if times is None:
                    times = h["time"]
                    speed = np.empty((flat_lat.size, len(times)))
                    direction = np.empty_like(speed)
                speed[s + k] = h["wind_speed_10m"]
                direction[s + k] = h["wind_direction_10m"]
            if ci < n_chunks - 1:
                time.sleep(1.0)  # be gentle with the free endpoint
        return speed, direction, times

    # ------------------------------------------------------------ interpolation
    def _cell(self, px, py):
        """Cell index and fractional position, clamped to the grid (edge extrapolation)."""
        fx = (px - self.x[0]) / self.dx
        fy = (py - self.y[0]) / self.dy
        i = np.clip(np.floor(fx), 0, self.x.size - 2).astype(np.int64)
        j = np.clip(np.floor(fy), 0, self.y.size - 2).astype(np.int64)
        tx = np.clip(fx - i, 0.0, 1.0)
        ty = np.clip(fy - j, 0.0, 1.0)
        return i, j, tx, ty

    def __call__(self, px, py):
        """Wind components at position(s). Accepts scalars or arrays; returns (wx, wy)."""
        px = np.asarray(px, dtype=np.float64)
        py = np.asarray(py, dtype=np.float64)
        i, j, tx, ty = self._cell(px, py)
        w00 = (1 - tx) * (1 - ty)
        w10 = tx * (1 - ty)
        w01 = (1 - tx) * ty
        w11 = tx * ty
        wx = (w00 * self.wx[i, j] + w10 * self.wx[i + 1, j]
              + w01 * self.wx[i, j + 1] + w11 * self.wx[i + 1, j + 1])
        wy = (w00 * self.wy[i, j] + w10 * self.wy[i + 1, j]
              + w01 * self.wy[i, j + 1] + w11 * self.wy[i + 1, j + 1])
        return wx, wy

    def torch_sampler(self, device="cpu", dtype=None):
        """Return a function (px, py) -> (wx, wy) operating on torch tensors."""
        import torch
        dtype = dtype or torch.float32
        key = (str(device), dtype)
        if key not in self._torch_cache:
            self._torch_cache[key] = (
                torch.as_tensor(self.wx, dtype=dtype, device=device),
                torch.as_tensor(self.wy, dtype=dtype, device=device),
            )
        twx, twy = self._torch_cache[key]
        x0, y0, dx, dy = self.x[0], self.y[0], self.dx, self.dy
        nx, ny = self.x.size, self.y.size

        def sample(px, py):
            fx = (px - x0) / dx
            fy = (py - y0) / dy
            i = torch.clamp(torch.floor(fx), 0, nx - 2).long()
            j = torch.clamp(torch.floor(fy), 0, ny - 2).long()
            tx = torch.clamp(fx - i, 0.0, 1.0)
            ty = torch.clamp(fy - j, 0.0, 1.0)
            w00 = (1 - tx) * (1 - ty)
            w10 = tx * (1 - ty)
            w01 = (1 - tx) * ty
            w11 = tx * ty
            wx = (w00 * twx[i, j] + w10 * twx[i + 1, j]
                  + w01 * twx[i, j + 1] + w11 * twx[i + 1, j + 1])
            wy = (w00 * twy[i, j] + w10 * twy[i + 1, j]
                  + w01 * twy[i, j + 1] + w11 * twy[i + 1, j + 1])
            return wx, wy

        return sample


# ---------------------------------------------------------------- generator
def generate_wind_field(
    rng=None,
    nx=101,
    ny=111,
    extent=(-1.0, 11.0),
    background_speed=(0.0, 4.0),
    gust_amplitude=(1.0, 4.0),
    gust_length=(0.5, 1.8),
    n_vortices=(0, 3),
    vortex_radius=(1.0, 3.0),
    vortex_strength=(2.0, 8.0),
    max_speed=10.0,
):
    """
    Sample a random smooth wind field on a grid over `extent` x `extent`.

    The field is the sum of
      * a uniform background wind with random direction and speed,
      * a smooth Gaussian random field (white noise filtered with a Gaussian
        kernel of random correlation length) for each component,
      * a random number of Gaussian vortices of random sign, radius, strength.
    Speeds above `max_speed` are rescaled to `max_speed`.

    `rng` can be None, an int seed, or a numpy Generator.
    """
    rng = np.random.default_rng(rng)
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[0], extent[1], ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = x[1] - x[0]

    # uniform background
    theta = rng.uniform(0.0, 2.0 * np.pi)
    u0 = rng.uniform(*background_speed)
    wx = np.full_like(X, u0 * np.cos(theta))
    wy = np.full_like(Y, u0 * np.sin(theta))

    # smooth random component
    length = rng.uniform(*gust_length)
    amp = rng.uniform(*gust_amplitude)
    sigma_px = length / dx
    for comp in (wx, wy):
        noise = gaussian_filter(rng.standard_normal(X.shape), sigma=sigma_px, mode="reflect")
        noise /= max(noise.std(), 1e-9)
        comp += amp * noise

    # Gaussian vortices: tangential speed peaks at rho = r with value `strength`
    for _ in range(rng.integers(n_vortices[0], n_vortices[1] + 1)):
        cx, cy = rng.uniform(extent[0], extent[1], size=2)
        r = rng.uniform(*vortex_radius)
        strength = rng.uniform(*vortex_strength) * rng.choice([-1.0, 1.0])
        px, py = X - cx, Y - cy
        rho = np.hypot(px, py) + 1e-9
        vt = strength * (rho / r) * np.exp(0.5 * (1.0 - (rho / r) ** 2))
        wx += vt * (-py / rho)
        wy += vt * (px / rho)

    # cap the speed
    speed = np.hypot(wx, wy)
    scale = np.where(speed > max_speed, max_speed / np.maximum(speed, 1e-9), 1.0)
    return WindField(x, y, wx * scale, wy * scale)


def uniform_wind_field(wx=0.0, wy=0.0, nx=101, ny=111, extent=(-1.0, 11.0)):
    """Constant wind field, useful for tests and sanity checks (zero wind by default)."""
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[0], extent[1], ny)
    return WindField(x, y, np.full((nx, ny), float(wx)), np.full((nx, ny), float(wy)))


def plot_wind_field(ax, wind, quiver_step=6, cmap="viridis", **quiver_kw):
    """Speed colormap plus direction quivers on a matplotlib axis. Returns the mesh."""
    im = ax.pcolormesh(wind.x, wind.y, wind.speed.T, shading="auto", cmap=cmap)
    X, Y = np.meshgrid(wind.x, wind.y, indexing="ij")
    s = quiver_step
    kw = dict(color="white", scale=150, width=0.003, alpha=0.8)
    kw.update(quiver_kw)
    ax.quiver(X[::s, ::s], Y[::s, ::s], wind.wx[::s, ::s], wind.wy[::s, ::s], **kw)
    ax.set_aspect("equal")
    return im
