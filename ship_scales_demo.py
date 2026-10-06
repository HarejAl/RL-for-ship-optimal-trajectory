"""
Same wind, same strategy, different ships: what the scaling says (PLAN.md step A2).

The wind is stored as a nondimensional field w~ = W / W_ref in [0, 1] (W_ref = 25 m/s), laid
over a 100 km box. Every ship follows the SAME naive strategy: full thrust pointed straight at
the destination. Ships differ only in their dimensional properties (mass, wetted area, hull
drag coefficient, windage area, air drag coefficient, service speed), which collapse into

    V* = sqrt(T / (0.5 rho_w C_w S))         calm-water speed at full thrust
    L* = 2 m / (rho_w C_w S)                 inertia length
    kappa = rho_a C_a A / (rho_w C_w S)      windage ratio
    A = kappa (W_ref / V*)^2                 wind authority: wind force on a stopped ship
                                             in a W_ref wind, over the maximum thrust

Figure 1 compares tracks, speed, wind and hull forces, voyage time and energy.
Figure 2 shows what each ship "sees" (the same map in units of its own V*) and the exact
collapse: a ship with more thrust in proportionally stronger wind sails the identical track.

Ship numbers are illustrative orders of magnitude, not data for any specific vessel. The model
is a point mass with isotropic quadratic drag: no heading-dependent windage, waves or currents.

    python ship_scales_demo.py
Outputs: output/ship_scales_compare.png, output/ship_scales_collapse.png
"""

import os
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wind import generate_wind_field
from dynamics import ShipParams

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
STYLE = os.path.join(SCRIPT_DIR, "journal.mplstyle")

RHO_W, RHO_A = 1025.0, 1.225
W_REF = 25.0            # m/s mapped to w~ = 1
KM_PER_UNIT = 10.0      # the generated field spans ~10 units -> 100 km
FIELD_SEED = 31
START_KM, GOAL_KM = np.array([10.0, 15.0]), np.array([90.0, 85.0])
ARRIVE_KM = 1.0


@dataclass(frozen=True)
class Ship:
    name: str
    mass: float         # kg (displacement)
    S: float            # wetted area, m^2
    Cw: float           # hull drag coefficient (on S)
    A: float            # windage (projected above-water) area, m^2
    Ca: float           # air drag coefficient (on A)
    v_service: float    # m/s, calm-water speed at full thrust (defines the thrust)
    colour: str

    @property
    def thrust(self):          # N, so that full thrust holds v_service in calm water
        return 0.5 * RHO_W * self.Cw * self.S * self.v_service ** 2

    @property
    def c_w(self):
        return 0.5 * RHO_W * self.Cw * self.S / self.mass

    @property
    def c_a(self):
        return 0.5 * RHO_A * self.Ca * self.A / self.mass

    @property
    def V(self):
        return np.sqrt(self.thrust / self.mass / self.c_w)

    @property
    def L(self):
        return 1.0 / self.c_w

    @property
    def kappa(self):
        return self.c_a / self.c_w

    @property
    def authority(self):
        return self.kappa * (W_REF / self.V) ** 2


SHIPS = [
    Ship("Container ship", 1.0e8, 12000, 0.0022, 1800, 0.8, 12.0, "#1f77b4"),
    Ship("Bulk carrier (loaded)", 2.0e8, 18000, 0.0020, 900, 0.8, 7.0, "#2ca02c"),
    Ship("Car carrier", 3.5e7, 8000, 0.0025, 3500, 0.9, 10.0, "#d62728"),
    Ship("Motor yacht 20 m", 4.0e4, 90, 0.0060, 45, 0.9, 6.0, "#ff7f0e"),
]


def wind_ms(field, x_km, y_km, k=1.0):
    """Wind in m/s at positions in km: w~ (field / 10) times W_ref, times an extra factor k."""
    wx, wy = field(x_km / KM_PER_UNIT, y_km / KM_PER_UNIT)
    return k * W_REF * wx / 10.0, k * W_REF * wy / 10.0


def sail(ship, field, wind_k=1.0, thrust_k=1.0):
    """Full thrust pointed at the goal. SI units throughout. Returns a dict of time series."""
    u_max = thrust_k * ship.thrust / ship.mass
    V = np.sqrt(u_max / ship.c_w)
    dt = (ship.L / V) / 10.0
    D = np.linalg.norm(GOAL_KM - START_KM) * 1e3
    t_max = 6.0 * D / V
    pos = START_KM * 1e3
    vel = np.zeros(2)
    rec = {k: [] for k in ("t", "x", "y", "sog", "hull", "wind", "progress")}
    t = 0.0
    arrived = False
    while t < t_max:
        to_goal = GOAL_KM * 1e3 - pos
        dist = np.linalg.norm(to_goal)
        if dist < ARRIVE_KM * 1e3:
            arrived = True
            break
        u = u_max * to_goal / dist
        wx, wy = wind_ms(field, pos[0] / 1e3, pos[1] / 1e3, wind_k)
        rv = vel - np.array([wx, wy])
        f_hull = -ship.c_w * np.linalg.norm(vel) * vel
        f_wind = -ship.c_a * np.linalg.norm(rv) * rv
        rec["t"].append(t)
        rec["x"].append(pos[0] / 1e3)
        rec["y"].append(pos[1] / 1e3)
        rec["sog"].append(np.linalg.norm(vel))
        rec["hull"].append(np.linalg.norm(f_hull) / u_max)
        rec["wind"].append(np.linalg.norm(f_wind) / u_max)
        rec["progress"].append(1.0 - dist / D)
        vel = vel + dt * (u + f_hull + f_wind)
        pos = pos + dt * vel
        t += dt
    out = {k: np.array(v) for k, v in rec.items()}
    out.update(arrived=arrived, T=t, V=V, D=D, dt=dt)
    return out


def cross_track(r):
    """Largest distance from the straight start-goal line, km."""
    d = (GOAL_KM - START_KM) / np.linalg.norm(GOAL_KM - START_KM)
    rel = np.c_[r["x"], r["y"]] - START_KM
    return float(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]).max())


def figure_compare(field, runs):
    fig = plt.figure(figsize=(16, 8.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    X, Y = np.meshgrid(field.x * KM_PER_UNIT, field.y * KM_PER_UNIT, indexing="ij")
    wt = field.speed / 10.0
    im = ax.pcolormesh(X, Y, wt, cmap="Blues", vmin=0, vmax=1, shading="gouraud")
    s = 6
    ax.quiver(X[::s, ::s], Y[::s, ::s], field.wx[::s, ::s], field.wy[::s, ::s], color="0.35",
              scale=200, width=0.002)
    ax.plot(*np.c_[START_KM, GOAL_KM], "k--", lw=1, label="straight line")
    for ship, r in zip(SHIPS, runs):
        ax.plot(r["x"], r["y"], color=ship.colour, lw=2.2,
                label=ship.name + ("" if r["arrived"] else "  (does not arrive)"))
    ax.plot(*START_KM, "ko", ms=7)
    ax.plot(*GOAL_KM, "k*", ms=13)
    ax.set(xlim=(0, 100), ylim=(0, 100), aspect="equal", xlabel="x [km]", ylabel="y [km]",
           title="Same wind, same strategy (full thrust at the goal)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, fraction=0.04, aspect=40)
    cb.set_label(r"nondimensional wind $\tilde w = |W| / W_{ref}$,   $W_{ref}$ = 25 m/s")

    panels = [
        (gs[0, 1], "sog", lambda sh, r: r["sog"] / sh.V, "speed over ground / V*"),
        (gs[0, 2], "wind", None, "wind force / max thrust"),
        (gs[1, 1], "hull", None, "hull drag / max thrust"),
    ]
    for spec, key, fn, lab in panels:
        a = fig.add_subplot(spec)
        for ship, r in zip(SHIPS, runs):
            y = fn(ship, r) if fn else r[key]
            a.plot(100 * r["progress"], y, color=ship.colour, lw=1.6)
        a.set(xlabel="progress along the line [%]", ylabel=lab, xlim=(0, 100))
        a.axhline(1.0, color="0.6", lw=0.8, ls=":")
    # summary bars
    a = fig.add_subplot(gs[1, 2])
    xs = np.arange(len(SHIPS))
    tr = np.array([r["T"] / (r["D"] / sh.V) for sh, r in zip(SHIPS, runs)])
    a.bar(xs, 100 * (tr - 1), 0.6, color=[s.colour for s in SHIPS])
    for x, sh, r in zip(xs, SHIPS, runs):
        a.text(x, 100 * (r["T"] / (r["D"] / sh.V) - 1) + 0.3, f"drift {cross_track(r):.1f} km",
               ha="center", va="bottom", fontsize=8)
    a.set_xticks(xs, [s.name.split(" (")[0].replace("Motor yacht 20 m", "Yacht") for s in SHIPS],
                 rotation=15, fontsize=8)
    a.set(ylabel="voyage time vs calm water [+%]", ylim=(0, 100 * (tr.max() - 1) * 1.45))
    fig.suptitle("Different ships in the same weather: the wind only matters through "
                 r"$\kappa$ and $W_{ref}/V^*$", fontsize=13)
    out = os.path.join(OUTPUT_DIR, "ship_scales_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_collapse(field, runs):
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.25]))
    X, Y = np.meshgrid(field.x * KM_PER_UNIT, field.y * KM_PER_UNIT, indexing="ij")
    vmax = 3.0
    for ax, ship in zip(axes[:4], SHIPS):
        seen = W_REF * field.speed / 10.0 / ship.V
        im = ax.pcolormesh(X, Y, seen, cmap="magma_r", vmin=0, vmax=vmax, shading="gouraud")
        ax.set(xlim=(0, 100), ylim=(0, 100), aspect="equal", xticks=[], yticks=[])
        ax.set_title(f"{ship.name}\n" + rf"$\kappa$={ship.kappa:.3f}  A={ship.authority:.2f}  "
                     rf"$L^*$={ship.L / 1e3:.1f} km", fontsize=9)
    cb = fig.colorbar(im, ax=axes[:4], orientation="horizontal", pad=0.04, fraction=0.06, aspect=60)
    cb.set_label(r"what the policy reads: $|W| / V^*$ (same map, each ship's own speed)")
    # exact collapse: container ship with 2x thrust (V* x sqrt 2) in sqrt(2) x stronger wind
    ax = axes[4]
    base = SHIPS[0]
    k = np.sqrt(2.0)
    r2 = sail(base, field, wind_k=k, thrust_k=2.0)
    r1 = runs[0]
    ax.plot(r1["x"], r1["y"], color=base.colour, lw=4, alpha=0.4, label="container ship, W")
    ax.plot(r2["x"], r2["y"], "k--", lw=1.2, label=r"2x thrust, $\sqrt{2}$x wind")
    ax.set(xlim=(0, 100), ylim=(0, 100), aspect="equal", xlabel="x [km]",
           title=f"Exact collapse: same track,\nvoyage {r1['T'] / 3600:.2f} h vs {r2['T'] / 3600:.2f} h "
                 rf"(ratio {r1['T'] / r2['T']:.3f} = $\sqrt{{2}}$)", )
    ax.legend(fontsize=8, loc="upper left")
    out = os.path.join(OUTPUT_DIR, "ship_scales_collapse.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(STYLE):
        plt.style.use(STYLE)
    field = generate_wind_field(FIELD_SEED)
    runs = [sail(s, field) for s in SHIPS]
    ref = ShipParams().scales()
    print(f"{'ship':24s} {'V* kn':>6s} {'L* km':>7s} {'T* s':>6s} {'kappa':>6s} {'Wref/V*':>7s} "
          f"{'A':>5s} {'D/L*':>6s} {'time/calm':>9s} {'drift km':>8s}")
    for s, r in zip(SHIPS, runs):
        print(f"{s.name:24s} {s.V / 0.5144:6.1f} {s.L / 1e3:7.2f} {s.L / s.V:6.0f} {s.kappa:6.3f} "
              f"{W_REF / s.V:7.2f} {s.authority:5.2f} {r['D'] / s.L:6.0f} "
              f"{r['T'] / (r['D'] / s.V):9.2f} {cross_track(r):8.1f}"
              + ("" if r["arrived"] else "  NOT ARRIVED"))
    # the RL model's reference ship, read at this map's scale (1 unit = 10 km, 2.5 m/s per unit)
    ms_per_unit = W_REF / 10.0
    V_ref = ref.speed * ms_per_unit
    print(f"{'RL model ship (toy)':24s} {V_ref / 0.5144:6.1f} {ref.length * KM_PER_UNIT:7.2f} "
          f"{ref.length * KM_PER_UNIT * 1e3 / V_ref:6.0f} {ref.windage:6.3f} {W_REF / V_ref:7.2f} "
          f"{ref.windage * (W_REF / V_ref) ** 2:5.2f} {113.0 / (ref.length * KM_PER_UNIT):6.0f}")
    print("saved", figure_compare(field, runs))
    print("saved", figure_collapse(field, runs))


if __name__ == "__main__":
    main()
