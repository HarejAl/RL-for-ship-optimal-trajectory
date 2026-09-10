"""
Hand-designed wind scenarios where the optimal route is VISIBLY non-trivial.

Real forecasts are smooth, so the optimal path is usually close to a straight line and a
picture of it teaches nothing. These scenarios put a structured feature between the port
and the destination -- a storm, a favourable jet, a barrier with a gap -- on top of a mild
Gaussian-random-field background, so the optimal trajectory has to do something a human
can immediately read: go around, detour into the fast lane, thread the gap.

Scale reminder (ShipParams): wind force is cd_air*|w|^2 = 0.25*w^2 and the per-axis thrust
limit is 10 (about 14 on the diagonal), so wind above ~7.5 units cannot be beaten head-on.
Blocking features are therefore built at 8-11 units (genuinely impassable, must be routed
around) and favourable jets at 5-7 units (worth a detour, not fatal).

Each builder returns (WindField, start, goal, title).
"""

import numpy as np

from wind import WindField, generate_wind_field

NX, NY = 121, 121
EXTENT = (-1.0, 11.0)


def _grid(nx=NX, ny=NY, extent=EXTENT):
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[0], extent[1], ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, X, Y


def _background(seed, amp=1.6, nx=NX, ny=NY, extent=EXTENT):
    """Mild GRF texture so the field looks like weather rather than a diagram."""
    wf = generate_wind_field(seed, nx=nx, ny=ny, extent=extent,
                             background_speed=(0.0, 1.2), gust_amplitude=(amp, amp),
                             gust_length=(1.2, 2.2), n_vortices=(0, 0), max_speed=99.0)
    return wf.wx.copy(), wf.wy.copy()


def _vortex(X, Y, cx, cy, r, strength):
    """Gaussian vortex: tangential speed peaks at radius r with magnitude |strength|."""
    px, py = X - cx, Y - cy
    rho = np.hypot(px, py) + 1e-9
    vt = strength * (rho / r) * np.exp(0.5 * (1.0 - (rho / r) ** 2))
    return vt * (-py / rho), vt * (px / rho)


def _jet(X, Y, p0, p1, width, strength):
    """A fast band of flow along p0->p1, Gaussian in the across-track direction."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.hypot(*d)
    t = d / L
    n = np.array([-t[1], t[0]])
    px, py = X - p0[0], Y - p0[1]
    across = px * n[0] + py * n[1]
    along = px * t[0] + py * t[1]
    prof = np.exp(-0.5 * (across / width) ** 2)
    prof *= np.clip(np.minimum(along, L - along) / (0.18 * L) + 1.0, 0.0, 1.0)  # fade at the ends
    return strength * prof * t[0], strength * prof * t[1]


def _band(X, Y, axis, centre, width, ux, uy):
    """A straight band of uniform flow across the domain ('x' = band along x, varying in y)."""
    coord = Y if axis == "x" else X
    prof = np.exp(-0.5 * ((coord - centre) / width) ** 2)
    return ux * prof, uy * prof


def storm_on_the_route(seed=11):
    """A cyclone squats on the direct line. The optimum must arc around its windward side."""
    x, y, X, Y = _grid()
    wx, wy = _background(seed)
    vx, vy = _vortex(X, Y, 5.0, 5.0, 2.1, 11.0)
    wx += vx
    wy += vy
    return (WindField(x, y, wx, wy), np.array([0.8, 0.8]), np.array([9.2, 9.2]),
            "Storm on the direct route")


def favourable_jet(seed=5):
    """A fast tailwind lane sits north of the rhumb line: worth a detour to ride it."""
    x, y, X, Y = _grid()
    wx, wy = _background(seed, amp=1.2)
    jx, jy = _jet(X, Y, (0.0, 7.2), (10.0, 7.2), width=1.30, strength=7.6)
    wx += jx
    wy += jy
    bx, by = _band(X, Y, "x", centre=3.6, width=1.8, ux=-4.6, uy=0.0)  # headwind along the direct line
    wx += bx
    wy += by
    return (WindField(x, y, wx, wy), np.array([0.6, 3.6]), np.array([9.4, 3.6]),
            "Favourable jet north of the rhumb line")


def barrier_with_a_gap(seed=23):
    """A wall of headwind spans the domain except for one narrow gap: thread it."""
    x, y, X, Y = _grid()
    wx, wy = _background(seed, amp=1.1)
    wall = np.exp(-0.5 * ((X - 5.0) / 0.9) ** 2)          # the wall, in x
    gap = 1.0 - np.exp(-0.5 * ((Y - 8.2) / 1.05) ** 2)    # punched through at y ~ 8.2
    wx += -10.5 * wall * gap
    wy += 2.0 * wall * gap * np.sign(Y - 8.2)             # splayed flow around the gap
    return (WindField(x, y, wx, wy), np.array([0.8, 3.0]), np.array([9.3, 6.0]),
            "Headwind barrier with one gap")


def twin_cyclones(seed=3):
    """Counter-rotating pair whose shared flow OPPOSES the direct line: go around one of them."""
    x, y, X, Y = _grid()
    wx, wy = _background(seed, amp=1.2)
    for (cx, cy, s) in ((4.6, 7.6, -10.0), (5.4, 2.4, 10.0)):
        vx, vy = _vortex(X, Y, cx, cy, 2.3, s)
        wx += vx
        wy += vy
    return (WindField(x, y, wx, wy), np.array([0.8, 5.0]), np.array([9.2, 5.0]),
            "Counter-rotating pair blocking the lane")


def shear_front(seed=31):
    """A sharp front: gale on one side, calm on the other. Hug the calm side, then cut across."""
    x, y, X, Y = _grid()
    wx, wy = _background(seed, amp=1.0)
    front = 1.0 / (1.0 + np.exp(-(Y - 5.6) * 2.6))  # smooth step in y
    wx += -9.5 * front
    wy += 2.2 * front
    return (WindField(x, y, wx, wy), np.array([0.8, 9.2]), np.array([9.2, 8.6]),
            "Shear front: gale above, calm below")


SCENARIOS = {
    "storm": storm_on_the_route,
    "jet": favourable_jet,
    "barrier": barrier_with_a_gap,
    "twin": twin_cyclones,
    "front": shear_front,
}


def build(name, **kw):
    return SCENARIOS[name](**kw)
