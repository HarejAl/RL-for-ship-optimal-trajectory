"""
Point-mass ship dynamics shared by the Gymnasium environment and the DP baseline.

Model (nondimensional, unit mass)
---------------------------------
    v_dot = u - c_w |v| v - c_a |v - W| (v - W)
    x_dot = v

    u : thrust (the control), clipped to |u_i| <= u_max
    W : wind velocity vector at the ship position
    c_w : hull drag coefficient (quadratic in velocity through the water, water at rest)
    c_a : wind drag coefficient (quadratic in velocity relative to the air)

Integration: semi-implicit (symplectic) Euler with step dt.

Stage cost
----------
    l(u) = dt * (time_w + ctrl_w * |u|^2)

The RL reward is minus this stage cost plus shaping terms, and the DP baseline
minimises exactly the sum of this stage cost, so both methods optimise the same
objective J = sum_k l(u_k).

All functions only use arithmetic operators, so they work unchanged with
Python floats, numpy arrays and torch tensors (batched or not).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShipParams:
    dt: float = 0.05          # integration step (20 Hz)
    cd_water: float = 0.5     # hull drag
    cd_air: float = 0.25      # wind drag
    u_max: float = 10.0       # per-axis thrust bound
    time_w: float = 1.0       # weight of travel time in the cost
    ctrl_w: float = 1e-2      # weight of control energy in the cost


def ship_accel(vx, vy, ux, uy, wx, wy, p: ShipParams):
    """Acceleration for velocity (vx, vy), thrust (ux, uy) and wind (wx, wy)."""
    speed = (vx * vx + vy * vy) ** 0.5
    rvx = vx - wx
    rvy = vy - wy
    rspeed = (rvx * rvx + rvy * rvy) ** 0.5
    ax = ux - p.cd_water * speed * vx - p.cd_air * rspeed * rvx
    ay = uy - p.cd_water * speed * vy - p.cd_air * rspeed * rvy
    return ax, ay


def ship_step(x, y, vx, vy, ux, uy, wx, wy, p: ShipParams):
    """One semi-implicit Euler step. Wind is evaluated at the current position."""
    ax, ay = ship_accel(vx, vy, ux, uy, wx, wy, p)
    vx1 = vx + p.dt * ax
    vy1 = vy + p.dt * ay
    x1 = x + p.dt * vx1
    y1 = y + p.dt * vy1
    return x1, y1, vx1, vy1


def stage_cost(ux, uy, p: ShipParams):
    """Running cost accumulated over one step: time plus control energy."""
    return p.dt * (p.time_w + p.ctrl_w * (ux * ux + uy * uy))


def terminal_speed(p: ShipParams, wind_speed: float) -> float:
    """
    Largest steady speed the ship can reach: full thrust along a tailwind of the
    given speed. Solves  c_w v^2 = u_max + c_a (w - v)^2  for v > 0.
    Used to size the velocity grid of the DP baseline.
    """
    u_eff = p.u_max * 2 ** 0.5  # per-axis bound allows |u| up to u_max*sqrt(2) on the diagonal
    a = p.cd_water - p.cd_air
    b = 2.0 * p.cd_air * wind_speed
    c = -(u_eff + p.cd_air * wind_speed ** 2)
    if abs(a) < 1e-12:
        return -c / b
    disc = b * b - 4.0 * a * c
    return (-b + disc ** 0.5) / (2.0 * a)
