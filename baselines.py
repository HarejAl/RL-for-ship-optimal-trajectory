"""
Naive reference controllers, used as a "do nothing clever" baseline for the optimal
(DP) and learned (DAgger) policies.

`straight_line_rollout` steers the ship straight at the goal with a saturated PID law
    u = clip( kp * (goal - pos) + ki * integral(goal - pos) - kd * v , -u_max, u_max )
ignoring the wind entirely (the wind still acts on the ship, so the realised path bends,
but the controller never plans around it). The integral term (with anti-windup) cancels
the steady-state offset a constant wind would otherwise leave, so the ship actually
reaches the small goal disc, as a real straight-line autopilot would. This is the "just
point at the destination and go" strategy a practitioner would use without any weather
routing, so the gap between it and DP is the value of route optimisation, and the gap
between it and the learned policy is what the policy actually delivers online.

Reported per rollout:
    t : travel time to the goal disc (s)
    E : control energy  = sum_k dt * |u_k|^2      (unweighted physical proxy)
    J : total optimised cost = sum_k dt*(time_w + ctrl_w |u_k|^2)   (matches the env)
"""

import numpy as np

from dynamics import ShipParams


def control_energy(actions, p: ShipParams):
    """Sum dt * |u|^2 over a trajectory (the physical control-effort proxy)."""
    actions = np.asarray(actions, dtype=np.float64)
    return float(p.dt * (actions ** 2).sum())


def straight_line_rollout(env, start, goal, wind, kp=4.0, ki=1.5, kd=4.0, i_max=8.0,
                          velocity=(0.0, 0.0), max_steps=None):
    """
    Roll out the wind-blind straight-line PID controller in `env`. Same return dict shape
    as ValueIterationPlanner.rollout, plus 'E' (control energy). `i_max` clamps the integral
    of the position error per axis (anti-windup).
    """
    p = env.p
    max_steps = max_steps or env.max_steps
    goal = np.asarray(goal, dtype=np.float64)
    obs, info = env.reset(options=dict(start=start, goal=goal, wind=wind, velocity=velocity))
    traj = [env.state.copy()]
    actions = []
    integ = np.zeros(2)
    terminated = truncated = False
    for _ in range(max_steps):
        x, y, vx, vy = env.state
        err = goal - np.array([x, y])
        integ = np.clip(integ + err * p.dt, -i_max, i_max)
        u = kp * err + ki * integ - kd * np.array([vx, vy])
        u = np.clip(u, -p.u_max, p.u_max)
        obs, r, terminated, truncated, info = env.step(u)
        traj.append(env.state.copy())
        actions.append(u)
        if terminated or truncated:
            break
    actions = np.array(actions)
    return dict(J=info["J"], t=info["t"], success=info["success"], oob=info["oob"],
                steps=len(actions), terminated=bool(terminated),
                traj=np.array(traj), actions=actions, E=control_energy(actions, p))
