"""
Isochrone weather routing -- the method used by practical sailing routers, and the fair
SPEED reference for a learned policy.

From the start, every `step_h` hours each point on the current front is pushed along a fan
of headings at its polar speed, giving a cloud of candidates. The cloud is pruned to its
outer boundary: bearings around the start are split into sectors and, on each tack, only the
candidate farthest from the start survives in each sector. That boundary is the isochrone -- the set of
places reachable in exactly t hours. The route is found when a candidate enters the goal
disc (the exact entry time along the last segment is solved for), then recovered by walking
parent pointers back to the start.

Tacks and gybes cost time: a candidate that changes side relative to the heading its parent
arrived on loses the penalty from its moving time; a penalty longer than one step is carried
over, so the time step and the penalty can be chosen independently. Time-varying weather is handled natively by
evaluating `wind_fn(t)` at each step -- one forward pass, no re-planning. That is the
property a DP formulation has to pay for with an extra time dimension.

Known approximations:
  * sector pruning assumes the reachable set is star-shaped around the start; large calm holes
    or obstacles can violate that and make the router sub-optimal;
  * pruning is greedy per step, so a manoeuvre penalty spanning more than ~2 steps can be pruned
    away before it pays off (a warning is raised). Keep step_h >= penalty / 2.
"""

import time

import numpy as np

from sailing.polar import wind_geometry, maneuver_kind


def _earliest_disc_entry(x, y, vx, vy, T, gx, gy, R):
    """Vectorised earliest tau in [0, T] with |p + v*tau - g| <= R; returns (hit, tau)."""
    px, py = x - gx, y - gy
    a = vx * vx + vy * vy
    b = 2.0 * (px * vx + py * vy)
    c = px * px + py * py - R * R
    disc = b * b - 4.0 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * np.maximum(a, 1e-15))
    inside = c <= 0.0
    hit = inside | ((disc >= 0.0) & (a > 1e-15) & (tau >= 0.0) & (tau <= T))
    return hit, np.where(inside, 0.0, np.clip(tau, 0.0, None))


def isochrone_route(polar, p, start, goal, wind=None, wind_fn=None, step_h=0.1, n_headings=72,
                    n_sectors=240, max_hours=200.0, keep_fronts=True):
    """
    Minimum-time route from `start` to `goal`.
    Returns dict(time_h, success, path (M,2), headings (M-1,), fronts, compute_s, tacks, gybes).
    """
    if wind is None and wind_fn is None:
        raise ValueError("provide wind or wind_fn")
    if max(p.tack_time, p.gybe_time) > 2.0 * step_h + 1e-12:
        import warnings
        warnings.warn(f"manoeuvre penalty ({max(p.tack_time, p.gybe_time)} h) spans more than 2 time steps "
                      f"(step_h={step_h} h): per-step pruning discards boats mid-manoeuvre before the tack pays "
                      f"off, and the router can fail to find a route. Validated up to 2 steps.")
    t_start = time.perf_counter()
    field0 = wind_fn(0.0) if wind_fn is not None else wind
    xmin, xmax, ymin, ymax = field0.extent
    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])
    H = np.linspace(-np.pi, np.pi, n_headings, endpoint=False)
    cosH, sinH = np.cos(H)[None, :], np.sin(H)[None, :]
    pen_of = np.array([0.0, p.tack_time, p.gybe_time])

    # `pend` = manoeuvre time still owed by that point: a tack longer than one step is served
    # across several steps instead of forcing step_h to exceed the penalty
    # `k` = index of the heading the point is sailing (-1 at the start).
    levels = [dict(x=np.array([sx]), y=np.array([sy]), h=np.array([np.nan]), parent=np.array([-1]),
                   pend=np.array([0.0]), k=np.array([-1]))]
    k_all = np.arange(n_headings)[None, :]
    t = 0.0
    arrival = None
    for step in range(int(np.ceil(max_hours / step_h))):
        L = levels[-1]
        field = wind_fn(t) if wind_fn is not None else wind
        wx, wy = field(L["x"], L["y"])
        wx, wy = np.asarray(wx, float), np.asarray(wy, float)

        tws, twa, rel_new = wind_geometry(wx[:, None], wy[:, None], H[None, :], p.kts_per_wind_unit)
        kts = polar.speed(twa, tws)
        v = kts / p.nm_per_unit
        has_h = ~np.isnan(L["h"])
        _, _, rel_old = wind_geometry(wx, wy, np.where(has_h, L["h"], 0.0), p.kts_per_wind_unit)
        kind = maneuver_kind(rel_old[:, None] * np.ones_like(rel_new), rel_new)
        pen = pen_of[kind] * has_h[:, None]
        owed = L["pend"][:, None] + pen
        eff = np.clip(step_h - owed, 0.0, None)
        pend_next = np.clip(owed - step_h, 0.0, None)
        vx, vy = v * cosH, v * sinH
        X0, Y0 = L["x"][:, None], L["y"][:, None]
        nx_, ny_ = X0 + vx * eff, Y0 + vy * eff
        # A boat still serving a manoeuvre is committed to its heading: it may not branch into a
        # new one until the tack/gybe is complete. Without this, every heading from a mid-tack
        # point is stationary at the same spot, they tie on distance, and the tie-break keeps an
        # arbitrary (often useless) heading -- which stalled the front whenever a tack spanned
        # more than one step.
        committed = L["pend"][:, None] > 1e-12
        allowed = ~committed | (k_all == L["k"][:, None])
        ok = (kts >= p.min_speed_kts) & allowed

        hit, tau = _earliest_disc_entry(X0, Y0, vx, vy, eff, gx, gy, p.goal_radius)
        hit &= ok
        if hit.any():
            t_hit = np.where(hit, t + np.minimum(owed, step_h) + tau, np.inf)
            fi, ki = np.unravel_index(int(np.argmin(t_hit)), t_hit.shape)
            arrival = dict(t=float(t_hit[fi, ki]), front_idx=int(fi), k=int(ki),
                           x=float(X0[fi, 0] + vx[fi, ki] * tau[fi, ki]),
                           y=float(Y0[fi, 0] + vy[fi, ki] * tau[fi, ki]))
            break

        ok &= (nx_ >= xmin) & (nx_ <= xmax) & (ny_ >= ymin) & (ny_ <= ymax)
        fi, ki = np.nonzero(ok)
        if fi.size == 0:
            break
        cx, cy = nx_[fi, ki], ny_[fi, ki]
        r = np.hypot(cx - sx, cy - sy)
        sector = np.floor((np.arctan2(cy - sy, cx - sx) + np.pi) / (2 * np.pi) * n_sectors).astype(np.int64)
        pend_c = pend_next[fi, ki]
        # Prune per (sector, tack side), not per sector. A boat that has just tacked has lost that
        # step's progress, so against same-sector neighbours still on the old tack it always
        # looks worse and is discarded before the tack can pay off -- which stalled the front on
        # any beat whose tack penalty exceeded the time step. Keeping the best boat on EACH tack
        # per sector lets the manoeuvre survive until its benefit shows.
        side = (rel_new[fi, ki] >= 0).astype(np.int64)
        group = sector * 2 + side
        order = np.lexsort((pend_c, -r, group))          # farthest first, least time owed on ties
        _, first = np.unique(group[order], return_index=True)
        keep = order[first]
        levels.append(dict(x=cx[keep], y=cy[keep], h=H[ki[keep]], parent=fi[keep],
                           pend=pend_c[keep], k=ki[keep]))
        t += step_h

    compute_s = time.perf_counter() - t_start
    fronts = [np.stack((lv["x"], lv["y"]), axis=1) for lv in levels] if keep_fronts else []
    if arrival is None:
        return dict(time_h=np.inf, success=False, path=np.array([[sx, sy]]), headings=np.array([]),
                    fronts=fronts, compute_s=compute_s, tacks=0, gybes=0)

    # walk parents back to the start
    pts = [(arrival["x"], arrival["y"])]
    heads = [H[arrival["k"]]]
    idx = arrival["front_idx"]
    for lv in reversed(levels):
        pts.append((lv["x"][idx], lv["y"][idx]))
        if lv["parent"][idx] < 0:
            break
        heads.append(lv["h"][idx])
        idx = lv["parent"][idx]
    path = np.array(pts[::-1])
    headings = np.array(heads[::-1])
    tacks, gybes = _count_manoeuvres(path, headings, p, wind if wind is not None else field0)
    return dict(time_h=arrival["t"], success=True, path=path, headings=headings, fronts=fronts,
                compute_s=compute_s, tacks=tacks, gybes=gybes)


def _count_manoeuvres(path, headings, p, wind):
    if len(headings) < 2:
        return 0, 0
    wx, wy = wind(path[1:-1, 0], path[1:-1, 1])
    _, _, r0 = wind_geometry(np.asarray(wx, float), np.asarray(wy, float), headings[:-1], p.kts_per_wind_unit)
    _, _, r1 = wind_geometry(np.asarray(wx, float), np.asarray(wy, float), headings[1:], p.kts_per_wind_unit)
    kind = maneuver_kind(r0, r1)
    return int((kind == 1).sum()), int((kind == 2).sum())


def route_follower(route, polar, p, gain=3.5, max_corr_deg=15.0):
    """
    Closed-loop policy that sails a planned route in `SailEnv`: hold the planned heading of the
    current leg, nudge back towards the leg line, and never let the nudge pinch the boat
    below 90% of the leg's polar speed (which would happen on a close-hauled leg).
    """
    path, heads = route["path"], route["headings"]
    state = {"i": 0}
    max_corr = np.deg2rad(max_corr_deg)

    def policy(env):
        x, y, _ = env.state
        pos = np.array([x, y])
        i = state["i"]
        while i < len(heads) - 1:
            a, b = path[i], path[i + 1]
            seg = b - a
            L2 = float(seg @ seg)
            if L2 < 1e-12 or float((pos - a) @ seg) / L2 >= 1.0:
                i += 1
            else:
                break
        state["i"] = i
        a, b = path[i], path[i + 1]
        seg = b - a
        n = np.array([-seg[1], seg[0]]) / max(np.linalg.norm(seg), 1e-12)
        h_seg = float(heads[i])
        h_cmd = h_seg - float(np.clip(gain * float((pos - a) @ n), -max_corr, max_corr))
        wx, wy = env.wind_at(env.t)(x, y)
        tws, twa_c, _ = wind_geometry(float(wx), float(wy), h_cmd, p.kts_per_wind_unit)
        _, twa_s, _ = wind_geometry(float(wx), float(wy), h_seg, p.kts_per_wind_unit)
        if polar.speed(twa_c, tws) < 0.9 * polar.speed(twa_s, tws):
            return h_seg
        return h_cmd

    return policy
