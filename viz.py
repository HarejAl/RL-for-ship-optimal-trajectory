"""
Shared visual style: a Windy-style wind colour scale.

NOTE ON PROVENANCE: windy.com does not publish its colour table in a fetchable form (the
site is a single-page app). What their documentation does state is that the wind overlay
follows the **Beaufort scale** up to 64 kt and NHC hurricane-category colours above it, and
the progression runs blue -> teal -> green -> olive/yellow -> orange -> red -> magenta.
The anchors below reproduce that progression and read the same way on a slide, but they are
an APPROXIMATION of Windy's palette, not the official table. Do not claim otherwise.

The anchors are tied to wind speed in m/s, so the same colour always means the same wind
strength regardless of the field being plotted -- which is what makes two maps comparable.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

# (m/s, colour) following the Beaufort progression Windy uses
WINDY_ANCHORS = [
    (0.0, "#6271b7"),   # calm: blue-violet
    (2.5, "#3961a0"),   # light air: blue
    (5.0, "#4a94a9"),   # light breeze: teal
    (7.5, "#4d8d7b"),   # gentle breeze: teal-green
    (10.0, "#53a553"),  # moderate breeze: green
    (12.5, "#359f35"),  # fresh breeze: green
    (15.0, "#a79d51"),  # strong breeze: olive/yellow
    (17.5, "#9f7f3a"),  # near gale: orange
    (20.0, "#a16c5c"),  # gale: orange-red
    (22.5, "#813a4e"),  # strong gale: dark red
    (25.0, "#af5088"),  # storm: magenta
]

WINDY_MAX_MS = WINDY_ANCHORS[-1][0]


def windy_cmap(name="windy", max_ms=WINDY_MAX_MS):
    """Windy-style colormap normalised over 0..max_ms (metres per second)."""
    stops = [(s / max_ms, c) for s, c in WINDY_ANCHORS if s <= max_ms]
    if stops[-1][0] < 1.0:
        stops.append((1.0, WINDY_ANCHORS[-1][1]))
    return LinearSegmentedColormap.from_list(name, stops)


def windy_norm(ms_per_unit=1.0, max_ms=WINDY_MAX_MS):
    """
    Normalize for a field stored in model units: colours stay anchored to real m/s.
    `ms_per_unit` comes from `field.meta` for real forecasts; use 1.0 to treat the field's
    own units as m/s (synthetic scenarios).
    """
    return Normalize(vmin=0.0, vmax=max_ms / max(ms_per_unit, 1e-9))


def wind_scale_ticks(ms_per_unit=1.0, max_ms=WINDY_MAX_MS, step=5.0):
    """Colorbar tick positions (in field units) and labels (in m/s)."""
    speeds = np.arange(0.0, max_ms + 1e-9, step)
    return speeds / max(ms_per_unit, 1e-9), [f"{s:.0f}" for s in speeds]


class WindParticles:
    """windy.com-style wind animation: a dense carpet of short, fine, faint strokes.

    Windy draws thousands of *short* streaks rather than long comet tails: each stroke covers a
    few frames of motion, is thin and semi-transparent, and particles are recycled constantly so
    the pattern keeps flowing. Shared by every animation in this repo, so the audience sees the
    same familiar picture each time.

        parts = WindParticles(ax, extent)
        parts.step(field)          # once per frame, with the wind field of that moment
    """

    def __init__(self, ax, extent, n=2600, tail=5, speed=0.30, lw=0.9, alpha=0.45, life=70,
                 colour="white", seed=3, zorder=1):
        from matplotlib.collections import LineCollection
        self.extent = tuple(float(v) for v in extent)
        self.n, self.tail, self.speed, self.life = int(n), int(tail), float(speed), int(life)
        self.rng = np.random.default_rng(seed)
        self.pts = self._spawn(self.n)
        self.age = self.rng.integers(0, self.life, self.n)
        self.hist = np.repeat(self.pts[None, :, :], self.tail, axis=0)     # (tail, n, 2)
        self.col = LineCollection([], colors=colour, linewidths=lw, alpha=alpha, zorder=zorder,
                                  capstyle="round")
        ax.add_collection(self.col)

    def _spawn(self, n):
        x0, x1, y0, y1 = self.extent
        return np.column_stack([self.rng.uniform(x0, x1, n), self.rng.uniform(y0, y1, n)])

    def step(self, field, dt=1.0):
        wx, wy = field(self.pts[:, 0], self.pts[:, 1])
        self.pts = self.pts + np.column_stack([wx, wy]) * self.speed * dt * 0.01
        self.age += 1
        x0, x1, y0, y1 = self.extent
        gone = ((self.pts[:, 0] < x0) | (self.pts[:, 0] > x1) | (self.pts[:, 1] < y0) |
                (self.pts[:, 1] > y1) | (self.age > self.life))
        if gone.any():
            idx = np.nonzero(gone)[0]
            self.pts[idx] = self._spawn(len(idx))
            self.age[idx] = 0
            self.hist[:, idx, :] = self.pts[idx]        # no streak across the respawn jump
        self.hist = np.roll(self.hist, -1, axis=0)
        self.hist[-1] = self.pts
        self.col.set_segments(np.transpose(self.hist, (1, 0, 2)))
        return self.col
