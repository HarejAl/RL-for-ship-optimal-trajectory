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

    def __init__(self, x, y, wx, wy):
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
    gust_length=(0.8, 3.0),
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
