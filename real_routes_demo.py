"""
The trained agent on REAL forecasts, at two very different scales (PLAN.md, deployment demo).

    container : a 14,000 TEU container ship, Lisbon -> Funchal (Madeira), ~510 nm
    yacht     : a 20 m motor yacht, Venice -> Rovinj, ~53 nm

The hourly Open-Meteo 10 m wind forecast (no API key) drives the ships from the current hour on,
evolving during the voyage. Ships sail their own physics in SI units. The agent (`ppo_pf_v1`,
trained on ONE nondimensional ship and synthetic fields only) is deployed purely by scaling:

    position : the route's box is mapped onto the agent's 10-unit world (km per unit differs
               by 10x between the two routes)
    speed    : ship velocity and wind are divided by the ship's own calm-water speed V*
    windage  : the wind it is shown is further scaled by sqrt(kappa / kappa_ref)
    time     : implicit; one decision every 0.05 model time units, which is 37 min for the
               container ship and 7.6 min for the yacht

Baselines, both pointed straight at the destination: "straight, economical" at the throttle that
minimises the agent's own cost (time + ctrl_w |u|^2) in calm water (58% of the thrust bound;
this isolates what ROUTING adds), and "straight, full thrust" (fastest, most fuel). The agent is trained to enter a
0.5-unit disc around the goal; inside it a final straight approach to the port takes over (the
same for both). The model has no land: tracks are drawn over the coastline but nothing avoids it.

    python real_routes_demo.py                     # both routes, departure = current UTC hour
    python real_routes_demo.py --routes yacht --depart-hour 6

Outputs: output/real_route_<name>.png, cached forecasts in output/cache/.
"""

import argparse
import datetime as dt
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics import ShipParams
from env import ShipEnv
from wind import WindField
from wind_obs import WindObsWrapper
from ppo_policy import PPOPolicy

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
STYLE = os.path.join(SCRIPT_DIR, "journal.mplstyle")

RHO_W, RHO_A = 1025.0, 1.225
KM_PER_DEG = 111.32
NM = 1.852
REF = ShipParams()
V_REF = REF.scales().speed          # agent's V* in model units
ECO_THROTTLE = float(np.sqrt(0.5 / (2.0 - 0.5) / REF.ctrl_w) / REF.u_max)  # argmin (1+c u^2)/sqrt(u/c_w)
ROUTE_UNITS = 8.0                   # the route spans this many model units (domain is -1..11)
FIELD_LO, FIELD_HI = -1.0, 11.0


@dataclass(frozen=True)
class Ship:
    name: str
    mass: float      # kg
    S: float         # wetted area, m^2
    Cw: float        # hull drag coefficient
    A: float         # windage area, m^2
    Ca: float        # air drag coefficient
    V: float         # calm-water speed at full thrust, m/s (per thrust axis, as in the model)

    @property
    def c_w(self):
        return 0.5 * RHO_W * self.Cw * self.S / self.mass

    @property
    def c_a(self):
        return 0.5 * RHO_A * self.Ca * self.A / self.mass

    @property
    def u_max(self):     # per-axis thrust bound, m/s^2
        return self.c_w * self.V ** 2

    @property
    def kappa(self):
        return self.c_a / self.c_w

    @property
    def L(self):
        return 1.0 / self.c_w


ROUTES = {
    "container": dict(
        ship=Ship("Container ship, 14,000 TEU", 1.9e8, 19000, 0.0021, 3000, 0.8, 12.0),
        start=("Lisbon", 38.62, -9.45), goal=("Funchal", 32.62, -16.92), colour="#00e5ff"),
    "yacht": dict(
        ship=Ship("Motor yacht, 20 m", 4.0e4, 90, 0.0060, 45, 0.9, 6.0),
        start=("Venice (Lido)", 45.42, 12.45), goal=("Rovinj", 45.08, 13.60), colour="#00e5ff"),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/ppo_pf_v1.pt")
    ap.add_argument("--routes", nargs="+", default=list(ROUTES), choices=list(ROUTES))
    ap.add_argument("--depart-hour", type=int, default=None,
                    help="forecast hour index to depart at (default: the current UTC hour today)")
    ap.add_argument("--grid", type=int, default=20, help="forecast points per side")
    ap.add_argument("--coast-grid", type=int, default=45, help="elevation points per side for the coastline")
    return ap.parse_args()


# ------------------------------------------------------------------ geography
class Geo:
    """Equirectangular km frame at the route's mean latitude, and the model-unit mapping."""

    def __init__(self, start, goal):
        self.lat0 = 0.5 * (start[1] + goal[1])
        self.lon0 = 0.5 * (start[2] + goal[2])
        self.kx = KM_PER_DEG * np.cos(np.deg2rad(self.lat0))
        s, g = self.km(start[1], start[2]), self.km(goal[1], goal[2])
        self.route_km = float(np.linalg.norm(g - s))
        self.kpu = self.route_km / ROUTE_UNITS               # km per model unit
        half = (FIELD_HI - FIELD_LO) / 2 * self.kpu
        self.lat_range = (self.lat0 - half / KM_PER_DEG, self.lat0 + half / KM_PER_DEG)
        self.lon_range = (self.lon0 - half / self.kx, self.lon0 + half / self.kx)

    def km(self, lat, lon):
        return np.array([(lon - self.lon0) * self.kx, (lat - self.lat0) * KM_PER_DEG])

    def unit(self, p_km):
        return np.asarray(p_km) / self.kpu + 5.0

    def lonlat(self, p_km):
        p = np.atleast_2d(p_km)
        return self.lon0 + p[:, 0] / self.kx, self.lat0 + p[:, 1] / KM_PER_DEG


def _get(url, params, tries=8):
    import requests
    for attempt in range(tries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            wait = min(10.0 * (attempt + 1), 60.0)
            print(f"  [open-meteo] {r.status_code}, retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def fetch_forecast(name, geo, n):
    """Hourly 10 m wind (east, north) in m/s on an n x n lat/lon grid, today + 2 days. Cached per day."""
    day = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    path = os.path.join(CACHE_DIR, f"real_route_{name}_{n}x{n}_{day}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d["u"], d["v"], list(d["times"])
    lats = np.linspace(*geo.lat_range, n)
    lons = np.linspace(*geo.lon_range, n)
    LO, LA = np.meshgrid(lons, lats, indexing="ij")
    flat_lat, flat_lon = LA.ravel(), LO.ravel()
    sp = dr = times = None
    for s in range(0, flat_lat.size, 200):
        sl = slice(s, s + 200)
        js = _get("https://api.open-meteo.com/v1/forecast", {
            "latitude": ",".join(f"{v:.4f}" for v in flat_lat[sl]),
            "longitude": ",".join(f"{v:.4f}" for v in flat_lon[sl]),
            "hourly": "wind_speed_10m,wind_direction_10m", "forecast_days": 3, "wind_speed_unit": "ms"})
        js = js if isinstance(js, list) else [js]
        for k, pt in enumerate(js):
            h = pt["hourly"]
            if times is None:
                times = h["time"]
                sp = np.empty((flat_lat.size, len(times)))
                dr = np.empty_like(sp)
            sp[s + k], dr[s + k] = h["wind_speed_10m"], h["wind_direction_10m"]
        time.sleep(1.0)
    u, v = WindField._uv(sp, dr)                      # (points, hours), east/north
    u = u.reshape(n, n, -1).transpose(2, 0, 1)        # (hours, nx=lon, ny=lat)
    v = v.reshape(n, n, -1).transpose(2, 0, 1)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, u=u, v=v, times=np.array(times))
    return u, v, times


def fetch_land(name, geo, n):
    """Elevation (m) on an n x n grid from the Open-Meteo elevation API; > 0 is land. Cached."""
    path = os.path.join(CACHE_DIR, f"real_route_{name}_elev{n}.npz")
    if os.path.exists(path):
        return np.load(path)["z"]
    lats = np.linspace(*geo.lat_range, n)
    lons = np.linspace(*geo.lon_range, n)
    LO, LA = np.meshgrid(lons, lats, indexing="ij")
    flat_lat, flat_lon = LA.ravel(), LO.ravel()
    z = np.empty(flat_lat.size)
    for s in range(0, flat_lat.size, 100):
        sl = slice(s, s + 100)
        js = _get("https://api.open-meteo.com/v1/elevation", {
            "latitude": ",".join(f"{v:.4f}" for v in flat_lat[sl]),
            "longitude": ",".join(f"{v:.4f}" for v in flat_lon[sl])})
        z[sl] = np.nan_to_num(np.array(js["elevation"], dtype=float))
        time.sleep(1.0)
    z = z.reshape(n, n)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, z=z)
    return z


# ----------------------------------------------------------------- simulation
class Weather:
    """True wind in m/s at (km position, seconds after departure): bilinear in space, linear in time."""

    def __init__(self, u, v, geo, h0):
        self.u, self.v, self.h0 = u, v, h0
        n = u.shape[1]
        half = (FIELD_HI - FIELD_LO) / 2 * geo.kpu
        self.axis = np.linspace(-half, half, n)       # km, same for x and y (square box)

    def field(self, t_s, scale=1.0, units=False, kpu=1.0):
        """WindField at time t (seconds from departure). units=True: axes in model units."""
        h = self.h0 + t_s / 3600.0
        i = int(np.clip(np.floor(h), 0, self.u.shape[0] - 2))
        a = float(np.clip(h - i, 0.0, 1.0))
        u = (1 - a) * self.u[i] + a * self.u[i + 1]
        v = (1 - a) * self.v[i] + a * self.v[i + 1]
        ax = self.axis / kpu + 5.0 if units else self.axis
        return WindField(ax, ax, scale * u, scale * v)


def sail(ship, weather, geo, start_km, goal_km, policy=None, throttle=None, port_km=1.0):
    """Simulate the real ship in SI units. policy=None: straight at the port, at `throttle`
    (fraction of the thrust bound) or, if None, at the most thrust the per-axis bound allows.
    Also integrates the agent's objective J = int (time_w + ctrl_w |u|^2) dt in model units."""
    v_unit = ship.V / V_REF                              # m/s per model velocity unit
    t_unit = geo.kpu * 1e3 / v_unit                      # seconds per model time unit
    decide_every = REF.dt * t_unit                       # seconds between decisions
    T_star = ship.L / ship.V
    n_sub = max(1, int(np.ceil(decide_every / (T_star / 10.0))))
    h = decide_every / n_sub
    perceive = np.sqrt(ship.kappa / 0.5) / v_unit        # m/s of true wind -> agent's wind units
    env = ShipEnv(weather.field(0.0, perceive, True, geo.kpu), params=REF)
    obs_wrap = WindObsWrapper(env, **{k: v for k, v in policy.obs_cfg.items() if k != "kappa_ref"}) \
        if policy is not None else None
    goal_u = geo.unit(goal_km)
    pos = np.array(start_km, dtype=float) * 1e3
    vel = np.zeros(2)
    t, mode = 0.0, "agent" if policy is not None else "straight"
    rec = dict(t=[0.0], x=[pos[0] / 1e3], y=[pos[1] / 1e3], sog=[0.0], mode=[mode], thrust=[0.0])
    energy = J = 0.0
    while t < 6.0 * geo.route_km * 1e3 / ship.V:
        p_km = pos / 1e3
        d_goal = np.linalg.norm(goal_km - p_km)
        if d_goal < port_km:
            break
        if mode == "agent" and d_goal / geo.kpu <= 0.5:
            mode = "final approach"
        if mode == "agent":
            env.wind = weather.field(t, perceive, True, geo.kpu)
            env.state = np.r_[geo.unit(p_km), vel / v_unit]
            env.goal = goal_u
            act, _ = policy.predict(obs_wrap.observation(None))
            frac = np.clip(np.asarray(act, float) / REF.u_max, -1, 1)
        arrived_now = False
        for _ in range(n_sub):
            to_port = goal_km - pos / 1e3
            if np.linalg.norm(to_port) < port_km:
                arrived_now = True
                break
            if mode == "agent" and np.linalg.norm(to_port) / geo.kpu <= 0.5:
                mode = "final approach"
            if mode != "agent":   # steered continuously: the most thrust the per-axis bound allows, at the port
                d = to_port / np.linalg.norm(to_port)
                if mode == "final approach":            # the agent's run ends like the eco baseline
                    frac = d * ECO_THROTTLE
                else:
                    frac = d * throttle if throttle is not None else d / np.abs(d).max()
            u = frac * ship.u_max
            wx, wy = weather.field(t)(pos[0] / 1e3, pos[1] / 1e3)
            rv = vel - np.array([float(wx), float(wy)])
            acc = u - ship.c_w * np.linalg.norm(vel) * vel - ship.c_a * np.linalg.norm(rv) * rv
            vel = vel + h * acc
            pos = pos + h * vel
            t += h
            energy += max(float(u @ vel), 0.0) * ship.mass * h
            J += (h / t_unit) * (REF.time_w + REF.ctrl_w * float(((frac * REF.u_max) ** 2).sum()))
            rec["t"].append(t)
            rec["x"].append(pos[0] / 1e3)
            rec["y"].append(pos[1] / 1e3)
            rec["sog"].append(np.linalg.norm(vel))
            rec["mode"].append(mode)
            rec["thrust"].append(float(np.linalg.norm(frac)))
        if arrived_now:
            break
    out = {k: np.array(v) for k, v in rec.items()}
    out.update(T=t, energy=energy, J=J, arrived=np.linalg.norm(goal_km - pos / 1e3) < port_km,
               decide_min=decide_every / 60.0, n_sub=n_sub)
    return out


# ---------------------------------------------------------------------- figure
def draw(name, cfg, geo, weather, land, times, h0, runs, ship):
    lons = np.linspace(*geo.lon_range, weather.u.shape[1])
    lats = np.linspace(*geo.lat_range, weather.u.shape[2])
    fig = plt.figure(figsize=(16, 8.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[:, 0])
    LO, LA = np.meshgrid(lons, lats, indexing="ij")
    f0 = weather.field(0.0)
    sp = np.hypot(f0.wx, f0.wy)
    im = ax.pcolormesh(LO, LA, sp, cmap="Blues", vmin=0, vmax=max(12.0, sp.max()), shading="gouraud")
    s = max(1, len(lons) // 14)
    ax.quiver(LO[::s, ::s], LA[::s, ::s], f0.wx[::s, ::s], f0.wy[::s, ::s], color="0.25", scale=260, width=0.0025)
    cl = np.linspace(*geo.lon_range, land.shape[0])
    ca = np.linspace(*geo.lat_range, land.shape[1])
    CLO, CLA = np.meshgrid(cl, ca, indexing="ij")
    ax.contourf(CLO, CLA, land, levels=[0.5, 1e5], colors=["#d9cbb0"], zorder=2)
    ax.contour(CLO, CLA, land, levels=[0.5], colors=["0.35"], linewidths=0.8, zorder=2)
    styles = {"agent": dict(color="#d6006f", lw=2.6, label="agent (ppo_pf_v1, scaled)"),
              "eco": dict(color="k", lw=1.6, ls="--", label="straight, economical throttle"),
              "full": dict(color="0.5", lw=1.2, ls=":", label="straight, full thrust")}
    for key, r in runs.items():
        lo, la = geo.lonlat(np.c_[r["x"], r["y"]])
        ax.plot(lo, la, zorder=4, **styles[key])
        every = 3 if name == "container" else 1        # hourly marks for the yacht, 3 h for the ship
        marks = [np.argmin(np.abs(r["t"] - k * every * 3600)) for k in range(1, int(r["T"] / 3600 / every) + 1)]
        ax.plot(lo[marks], la[marks], "o", ms=3.5, color=styles[key]["color"], zorder=5)
    for (lab, la, lo), mk in ((cfg["start"], "o"), (cfg["goal"], "*")):
        ax.plot(lo, la, mk, color="k", ms=11 if mk == "*" else 8, zorder=6)
        ax.annotate(lab, (lo, la), xytext=(6, 6), textcoords="offset points", fontsize=10, zorder=6)
    ax.set(xlim=geo.lon_range, ylim=geo.lat_range, xlabel="longitude", ylabel="latitude",
           aspect=1.0 / np.cos(np.deg2rad(geo.lat0)))
    dep = times[h0] if h0 < len(times) else "?"
    ax.set_title(f"{ship.name}: {cfg['start'][0]} -> {cfg['goal'][0]}, {geo.route_km / NM:.0f} nm\n"
                 f"Open-Meteo 10 m wind at departure ({dep} UTC); dots every "
                 f"{3 if name == 'container' else 1} h", fontsize=11)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.01)
    cb.set_label("wind speed at departure [m/s]")

    a = fig.add_subplot(gs[0, 1])
    for key, r in runs.items():
        a.plot(r["t"] / 3600, r["sog"] / 0.5144, color=styles[key]["color"], lw=1.5,
               ls=styles[key].get("ls", "-"))
        w = [np.hypot(*map(float, weather.field(t)(x, y))) for t, x, y in zip(r["t"], r["x"], r["y"])]
        if key == "agent":
            a.plot(r["t"] / 3600, np.array(w) / 0.5144, color="#4a90c2", lw=1, alpha=0.8,
                   label="wind speed at the agent")
    a.axhline(ship.V / 0.5144, color="0.6", ls=":", lw=0.8)
    a.set(xlabel="hours since departure", ylabel="knots", title="speed over ground (and wind at the agent)")
    a.legend(fontsize=8)

    a = fig.add_subplot(gs[1, 1])
    a.axis("off")
    ra, re, rf = runs["agent"], runs["eco"], runs["full"]
    v_unit = ship.V / V_REF
    lines = [
        f"ship: V* = {ship.V / 0.5144:.1f} kn,  kappa = {ship.kappa:.3f},  L* = {ship.L / 1e3:.2f} km",
        f"wind authority A = kappa (25 m/s / V*)^2 = {ship.kappa * (25 / ship.V) ** 2:.2f}",
        "",
        "deployment scaling (no retraining):",
        f"  1 model unit = {geo.kpu:.1f} km;  1 speed unit = {v_unit:.2f} m/s",
        f"  wind shown x sqrt(kappa/0.5) = x{np.sqrt(ship.kappa / 0.5):.2f}",
        f"  one decision every {ra['decide_min']:.1f} min",
        "",
        f"{'':20s}{'agent':>9s}{'straight':>10s}{'straight':>10s}",
        f"{'':20s}{'':>9s}{'eco':>10s}{'full':>10s}",
        f"{'voyage time [h]':20s}{ra['T'] / 3600:9.2f}{re['T'] / 3600:10.2f}{rf['T'] / 3600:10.2f}",
        f"{'energy (vs eco)':20s}{ra['energy'] / re['energy']:9.2f}{1.0:10.2f}{rf['energy'] / re['energy']:10.2f}",
        f"{'cost J (objective)':20s}{ra['J']:9.2f}{re['J']:10.2f}{rf['J']:10.2f}",
        "",
        f"agent vs straight-eco: J {100 * (ra['J'] / re['J'] - 1):+.1f}%, time "
        f"{100 * (ra['T'] / re['T'] - 1):+.1f}%, energy {100 * (ra['energy'] / re['energy'] - 1):+.1f}%",
    ]
    a.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=10)
    out = os.path.join(OUTPUT_DIR, f"real_route_{name}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(STYLE):
        plt.style.use(STYLE)
    policy = PPOPolicy(os.path.join(SCRIPT_DIR, args.model))
    h0 = args.depart_hour if args.depart_hour is not None else dt.datetime.now(dt.timezone.utc).hour
    for name in args.routes:
        cfg = ROUTES[name]
        ship = cfg["ship"]
        geo = Geo(cfg["start"], cfg["goal"])
        u, v, times = fetch_forecast(name, geo, args.grid)
        land = fetch_land(name, geo, args.coast_grid)
        weather = Weather(u, v, geo, h0)
        s_km, g_km = geo.km(*cfg["start"][1:]), geo.km(*cfg["goal"][1:])
        runs = {"agent": sail(ship, weather, geo, s_km, g_km, policy),
                "eco": sail(ship, weather, geo, s_km, g_km, None, throttle=ECO_THROTTLE),
                "full": sail(ship, weather, geo, s_km, g_km, None)}
        ra = runs["agent"]
        print(f"{name}: {geo.route_km / NM:.0f} nm, {geo.kpu:.1f} km/unit, decision every {ra['decide_min']:.1f} min, "
              f"{ra['n_sub']} physics substeps; kappa {ship.kappa:.3f}; departure {times[h0]} UTC")
        for k, r in runs.items():
            print(f"   {k:6s} {r['T'] / 3600:6.2f} h  J {r['J']:6.3f}  energy x{r['energy'] / runs['eco']['energy']:.3f}"
                  f"  arrived={r['arrived']}")
        print("   saved", draw(name, cfg, geo, weather, land, times, h0, runs, ship))


if __name__ == "__main__":
    main()
