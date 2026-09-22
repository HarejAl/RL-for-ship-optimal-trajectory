"""
Regatta: every sailing agent we trained, plus DP and a naive helmsman, race the same course.

    python sailing/regatta.py --seed 0            # one race in evolving wind -> GIF + final standings
    python sailing/regatta.py --seed 0 --wind static   # frozen wind map instead
    python sailing/regatta.py --series 5          # 5 races (seeds 0..4), points table, no GIFs
    python sailing/regatta.py --seed 4 --anon     # boats labelled "Agent 1..7"; key printed to console

Course (windward-leeward): start -> windward mark (upwind) -> leeward mark -> finish.
Wind (--wind evolving, default): a northerly that slowly veers or backs by 12-20 deg over the race,
a band of stronger breeze drifting across the course, the breeze building, and gust patches that
morph smoothly (all on time scales of a day or more; slices every 2 h, blended in time). --wind static: one frozen gusty map with more pressure on one side.
Every boat feels the wind at its own position and time.

Fleet
--fleet algos races one boat per RL algorithm (PPO / A2C / DQN / SAC), all trained on the same
regatta winds, against DP and a naive helmsman.

    The Pro        DP, one solve per mark; in evolving wind the time-dependent DP that knows
                   the whole forecast (a perfect-forecast navigator)
    Veteran        PPO, pure RL, final model (3M steps)
    Hothead        PPO, the checkpoint with 100% held-out arrivals but more tacks
    Rookie         PPO, earlier snapshot (~1.6M steps)
    Copycat        DP-taught clone (DAgger), trained on random wind maps
    Chatterbox     SAC with continuous heading (never learned to stop tacking)
    Tourist        points the bow at the next mark, always

A boat that leaves the map is out (DNF "aground"); the race closes after --max-hours.
Scoring for --series: low-point system, 1 point for 1st ... DNF = fleet size + 1.
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, REPO_DIR)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrow
from scipy.ndimage import gaussian_filter

from wind import WindField
from viz import windy_cmap, windy_norm
from sailing.rl_env import SailRLEnv, RL_PARAMS
from sailing.dp_sail import SailDP

MODEL_DIR = os.path.join(REPO_DIR, "models", "sail")
OUT_DIR = os.path.join(REPO_DIR, "output", "sailing", "rl")
BG = "#04121f"
KNOT_MS = 0.514444

# short course: ~45 nm beat, ~30 nm run, ~15 nm reach to the finish
VIEW = (1.2, 8.8, 0.9, 8.1)          # plotted area (xmin, xmax, ymin, ymax); boats may sail outside it
MARKS = dict(start=(5.0, 2.2), windward=(5.0, 6.8), leeward=(4.0, 3.4), finish=(5.8, 2.4))
COURSE = ("windward", "leeward", "finish")


# ------------------------------------------------------------------ wind
def race_wind(seed, n=121, extent=(-1.0, 11.0), base_kts=12.0):
    """Northerly with gusts and a pressure gradient across the course (side chosen by seed)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(*extent, n)
    y = np.linspace(*extent, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    base = base_kts / RL_PARAMS.kts_per_wind_unit
    side = rng.choice([-1.0, 1.0])
    pressure = 1.0 + 0.22 * side * np.tanh((X - 5.0) / 2.5)            # stronger on one side
    dx = x[1] - x[0]
    gust = [gaussian_filter(rng.standard_normal(X.shape), 1.4 / dx, mode="reflect") for _ in range(2)]
    gust = [g / g.std() for g in gust]
    shift = np.deg2rad(rng.uniform(-8, 8)) + np.deg2rad(7) * gust[0]     # direction wobble
    speed = base * pressure * (1.0 + 0.15 * gust[1])
    wx = speed * np.sin(shift)
    wy = -speed * np.cos(shift)
    return WindField(x, y, wx, wy), ("right" if side > 0 else "left")


def race_wind_seq(seed, horizon_h=48.0, slice_h=2.0, n=81, extent=(-1.0, 11.0)):
    """Slowly evolving weather: a persistent shift, a drifting pressure band, a building breeze,
    and gust patches that morph smoothly (no flicker between slices)."""
    from sailing.wind_seq import WindSequence
    rng = np.random.default_rng(1000 + seed)
    x = y = np.linspace(*extent, n)
    X, _ = np.meshgrid(x, y, indexing="ij")
    dx = x[1] - x[0]

    def patches():
        return [gaussian_filter(rng.standard_normal(X.shape), 1.6 / dx, mode="reflect") for _ in range(2)]

    A, B = patches(), patches()
    A, B = [a / a.std() for a in A], [b / b.std() for b in B]
    shift = np.deg2rad(rng.uniform(12, 20)) * rng.choice([-1.0, 1.0])     # total veer/back over the race
    shift0 = np.deg2rad(rng.uniform(-6, 6))
    band_x0, band_speed = rng.uniform(1.0, 9.0), rng.uniform(-0.12, 0.12)   # units per hour
    morph = 2 * np.pi / rng.uniform(40, 70)                                 # gust pattern period
    kts0, kts1 = rng.uniform(9, 11), rng.uniform(13, 15)
    times, fields = [], []
    for t in np.arange(0.0, horizon_h + 1e-9, slice_h):
        g = [np.cos(morph * t) * a + np.sin(morph * t) * b for a, b in zip(A, B)]   # unit variance
        u = min(t / 30.0, 1.0)
        veer = shift0 + shift * (3 * u ** 2 - 2 * u ** 3) + np.deg2rad(5) * g[0]
        cx = np.clip(band_x0 + band_speed * t, -1.0, 11.0)
        kts = kts0 + (kts1 - kts0) * u
        speed = (kts / RL_PARAMS.kts_per_wind_unit * (0.85 + 0.35 * np.exp(-(X - cx) ** 2 / (2 * 2.2 ** 2)))
                 * (1.0 + 0.15 * g[1]))
        fields.append(WindField(x, y, speed * np.sin(veer), -speed * np.cos(veer)))
        times.append(t)
    turn = "veering" if shift < 0 else "backing"
    desc = f"{turn} {np.rad2deg(abs(shift)):.0f} deg, building {kts0:.0f} -> {kts1:.0f} kt"
    return WindSequence(times, fields), desc


# ------------------------------------------------------------------ fleet
def load_fleet(kind="all", map_model="ppo_regatta_map.zip", blind_model="ppo_regatta_blind.zip"):
    """kind='all': every agent. kind='map-vs-blind': the two regatta-trained PPO agents, identical
    except that one reads the CNN wind map, with DP as the reference boat."""
    from stable_baselines3 import PPO, SAC
    from sailing.rl_eval import sb3_policy
    from sailing.clone_dp import Student, student_policy, MAP_RES

    fleet = []

    def add(name, color, env, make_policy):
        fleet.append(dict(name=name, color=color, env=env, make_policy=make_policy))

    add("The Pro (DP)", "#ffffff", SailRLEnv(None), "dp")
    if kind == "algos":
        # one boat per RL algorithm, all trained on the same regatta winds with the same
        # time-to-go shaping; plus the older PPO trained with plain distance shaping
        from sailing.rl_eval import load_agent
        for name, file, color in (
                ("PPO", "ppo_time_best.zip", "#00e5ff"),
                ("A2C", "a2c_time_best.zip", "#b8f35f"),
                ("DQN", "dqn_time_best.zip", "#ffd166"),
                ("SAC (continuous)", "sac_time_best.zip", "#ff8fab"),
                ("PPO, distance shaping", "ppo_regatta_blind.zip", "#c792ea")):
            path = os.path.join(MODEL_DIR, file)
            if not os.path.exists(path):
                print(f"  (skipping {name}: {file} not found)")
                continue
            pol, env = load_agent(path)
            add(name, color, env, lambda mark, p=pol: p)
        add("Tourist (aims at mark)", "#ff9f1c", SailRLEnv(None), lambda mark: (lambda env: env.heading_for(0.0)))
        return fleet
    if kind == "map-vs-blind":
        mpol = sb3_policy(PPO.load(os.path.join(MODEL_DIR, map_model), device="cpu"))
        add("MAP (PPO + CNN map)", "#00e5ff", SailRLEnv(None, map_res=16, n_actions=36), lambda mark: mpol)
        bpol = sb3_policy(PPO.load(os.path.join(MODEL_DIR, blind_model), device="cpu"))
        add("BLIND (PPO, local wind)", "#ff9f1c", SailRLEnv(None, n_actions=36), lambda mark: bpol)
        return fleet
    for name, file, color in (("Veteran (PPO 3M)", "ppo_uniform.zip", "#00e5ff"),
                              ("Hothead (PPO best)", "ppo_uniform_best.zip", "#ffd166"),
                              ("Rookie (PPO 1.6M)", "ppo_uniform_snapshot.zip", "#b8f35f")):
        pol = sb3_policy(PPO.load(os.path.join(MODEL_DIR, file), device="cpu"))
        add(name, color, SailRLEnv(None, n_actions=36), lambda mark, p=pol: p)
    net = Student()
    net.load_state_dict(torch.load(os.path.join(MODEL_DIR, "clone_fields_r3.pt"), map_location="cpu"))
    net.eval()
    cpol = student_policy(net, "cpu")
    add("Copycat (DP clone)", "#c792ea", SailRLEnv(None, map_res=MAP_RES), lambda mark: cpol)
    spol = sb3_policy(SAC.load(os.path.join(MODEL_DIR, "sac_uniform_best.zip"), device="cpu"))
    add("Chatterbox (SAC)", "#ff8fab", SailRLEnv(None), lambda mark: spol)
    add("Tourist (aims at mark)", "#ff9f1c", SailRLEnv(None), lambda mark: (lambda env: env.heading_for(0.0)))
    return fleet


# ------------------------------------------------------------------ race
def race(fleet, wind, max_hours=45.0):
    """`wind`: a static WindField or a WindSequence (evolving)."""
    dt = RL_PARAMS.dt
    seq = wind if hasattr(wind, "fields") else None
    field0 = seq.at(0.0) if seq is not None else wind
    marks = [np.array(MARKS[m]) for m in COURSE]
    dp_solvers = {}
    polar = fleet[0]["env"].sail.polar
    for m in COURSE:
        if seq is not None:
            from sailing.dp_time import SailDPTime
            dp = SailDPTime(seq, polar, RL_PARAMS, np.array(MARKS[m]), nx=81, ny=81)
        else:
            dp = SailDP(wind, polar, RL_PARAMS, np.array(MARKS[m]), nx=101, ny=101)
        dp.solve()
        dp_solvers[m] = dp.policy()

    boats = []
    for f in fleet:
        env = f["env"]
        env.reset(options=dict(wind=field0, start=np.array(MARKS["start"]), goal=marks[0],
                               heading=np.pi / 2))
        env.sail.wind_fn = seq
        boats.append(dict(f, leg=0, t_finish=None, status="racing", traj=[env.sail.state.copy()],
                          tacks=[0], rounds=[]))

    def policy_for(b):
        mark = COURSE[b["leg"]]
        if b["make_policy"] == "dp":
            pol = dp_solvers[mark]
            return lambda env: pol(env.sail)
        return b["make_policy"](mark)

    n_steps = int(round(max_hours / dt))
    for step in range(n_steps):
        live = [b for b in boats if b["status"] == "racing"]
        if not live:
            break
        for b in boats:
            env = b["env"]
            if b["status"] == "racing":
                h = policy_for(b)(env)
                _, _, term, trunc, info = env.sail.step(np.array([h]))
                if info["success"]:
                    b["rounds"].append((len(b["traj"]), env.sail.t))
                    b["leg"] += 1
                    if b["leg"] == len(COURSE):
                        b["status"], b["t_finish"] = "finished", env.sail.t
                    else:
                        env.sail.goal = marks[b["leg"]]
                elif info["oob"]:
                    b["status"] = "aground"
                env.sail.steps = min(env.sail.steps, 1)          # no per-leg step limit
            b["traj"].append(env.sail.state.copy())
            b["tacks"].append(env.sail.tacks + env.sail.gybes)
    for b in boats:
        if b["status"] == "racing":
            b["status"] = "time limit"
        b["traj"], b["tacks"] = np.array(b["traj"]), np.array(b["tacks"])
    return boats


def remaining_nm(b, k=None):
    """Distance still to sail along the course (rhumb lines), for ranking boats still racing."""
    k = len(b["traj"]) - 1 if k is None else k
    leg = sum(1 for i, _ in b["rounds"] if i <= k)
    if leg >= len(COURSE):
        return 0.0
    pos = b["traj"][k, :2]
    pts = [MARKS[m] for m in COURSE[leg:]]
    d = np.hypot(*(np.array(pts[0]) - pos))
    for a, c in zip(pts[:-1], pts[1:]):
        d += np.hypot(*(np.array(c) - np.array(a)))
    return d * RL_PARAMS.nm_per_unit


def standings(boats):
    fin = sorted([b for b in boats if b["status"] == "finished"], key=lambda b: b["t_finish"])
    rest = sorted([b for b in boats if b["status"] != "finished"], key=lambda b: remaining_nm(b))
    return fin + rest


# ------------------------------------------------------------------ animation
def animate(boats, wind, title, seed, fps=15, stride=2, tag=""):
    p = RL_PARAMS
    seq = wind if hasattr(wind, "fields") else None
    wind = seq.at(0.0) if seq is not None else wind
    xmin, xmax, ymin, ymax = wind.extent
    fig = plt.figure(figsize=(11.5, 8.0), facecolor=BG)
    ax = fig.add_axes([0.02, 0.04, 0.62, 0.88])
    board = fig.add_axes([0.66, 0.40, 0.32, 0.52])
    board.axis("off")
    prog = fig.add_axes([0.71, 0.07, 0.26, 0.24])     # race progress: distance still to sail
    prog.set_facecolor(BG)
    prog.set_title("distance still to sail (nm)", color="white", fontsize=9.5, pad=4)
    prog.tick_params(colors="white", labelsize=8)
    prog.set_xlabel("hours", color="white", fontsize=8.5, labelpad=1)
    prog.grid(True, color="#1e3a4f", lw=0.6, alpha=0.8)
    for sp in prog.spines.values():
        sp.set_color("#2a4a63")
    ax.set_facecolor(BG)
    mesh = ax.pcolormesh(wind.x, wind.y, wind.speed.T, shading="gouraud", cmap=windy_cmap(),
                  norm=windy_norm(p.kts_per_wind_unit * KNOT_MS), zorder=0)
    xmin, xmax, ymin, ymax = VIEW                  # zoom on the race area
    X, Y = np.meshgrid(np.linspace(xmin + 0.3, xmax - 0.3, 12), np.linspace(ymin + 0.3, ymax - 0.3, 12), indexing="ij")
    wx, wy = wind(X, Y)
    quiv = ax.quiver(X, Y, wx, wy, color="white", alpha=0.45, scale=110, width=0.004, zorder=1)
    for name in COURSE:
        mx, my = MARKS[name]
        ax.plot(mx, my, "o", color="#ff595e" if name != "finish" else "#8ac926", mec="white", mew=1.5,
                ms=13, zorder=6)
        ax.text(mx + 0.3, my + 0.25, name, color="white", fontsize=9, zorder=6,
                path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
    ax.plot(*MARKS["start"], "s", color="white", ms=8, zorder=6)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal", xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_color("#2a4a63")
    ax.set_title(f"Regatta #{seed}  -  {title}", color="white", fontsize=13)

    lines, arrows = [], [None] * len(boats)
    for b in boats:
        (ln,) = ax.plot([], [], color=b["color"], lw=1.8, alpha=0.9, zorder=4,
                        path_effects=[pe.Stroke(linewidth=3.5, foreground=BG, alpha=0.5), pe.Normal()])
        lines.append(ln)
    board.text(0.0, 0.98, "LEADERBOARD", color="white", fontsize=14, weight="bold", family="monospace",
               transform=board.transAxes, va="top")
    clock = board.text(0.0, 0.925, "", color="#9ad1ff", fontsize=10.5, family="monospace",
                       transform=board.transAxes, va="top")
    rows = [board.text(0.0, 0.80 - 0.135 * i, "", fontsize=10.5, family="monospace",
                       transform=board.transAxes, va="top") for i in range(len(boats))]
    # remaining distance per boat, precomputed once so the progress chart is cheap to draw
    remain = np.array([[remaining_nm(b, k) for k in range(len(b["traj"]))] for b in boats], dtype=object)
    plines = [prog.plot([], [], color=b["color"], lw=1.8)[0] for b in boats]

    n = max(len(b["traj"]) for b in boats)
    fin = [b["rounds"][-1][0] for b in boats if b["status"] == "finished"]
    if fin:                                    # stop a few hours after the last finisher
        n = min(n, max(fin) + int(2.0 / p.dt))
    frames = list(range(0, n, stride)) + [n - 1] * (3 * fps)
    leg_names = ["-> windward", "-> leeward", "-> finish"]

    def update(k):
        for i, b in enumerate(boats):
            j = min(k, len(b["traj"]) - 1)
            tr = b["traj"]
            lines[i].set_data(tr[max(0, j - 400):j + 1, 0], tr[max(0, j - 400):j + 1, 1])
            if arrows[i] is not None:
                arrows[i].remove()
            x, y, h = tr[j]
            arrows[i] = ax.add_patch(FancyArrow(x - 0.2 * np.cos(h), y - 0.2 * np.sin(h), 0.4 * np.cos(h),
                                                0.4 * np.sin(h), width=0.1, head_width=0.32, head_length=0.24,
                                                color=b["color"], ec=BG, lw=0.8, zorder=8,
                                                length_includes_head=True))
        t = k * p.dt
        f = seq.at(t) if seq is not None else wind
        if seq is not None:
            mesh.set_array(f.speed.T.ravel())
            quiv.set_UVC(*f(X, Y))
        cwx, cwy = f(5.0, 5.0)
        wfrom = (np.rad2deg(np.arctan2(-float(cwx), -float(cwy))) + 360) % 360
        for i, b in enumerate(boats):
            j = min(k, len(b["traj"]) - 1)
            plines[i].set_data(np.arange(j + 1) * p.dt, remain[i][:j + 1])
        prog.set_xlim(0, max(1.0, k * p.dt))
        prog.set_ylim(0, max(float(np.max(r)) for r in remain) * 1.05)

        clock.set_text(f"race clock  {t:5.1f} h\nwind mid-course: from {wfrom:03.0f} deg, "
                       f"{np.hypot(float(cwx), float(cwy)) * p.kts_per_wind_unit:4.1f} kt")

        def key(b):
            j = min(k, len(b["traj"]) - 1)
            done = sum(1 for i, _ in b["rounds"] if i <= j)
            if done == len(COURSE):
                return (0, b["t_finish"])
            if b["status"] == "aground" and j < k:
                return (2, 0)
            return (1, remaining_nm(b, j))

        for r, b in zip(rows, sorted(boats, key=key)):
            j = min(k, len(b["traj"]) - 1)
            done = sum(1 for i, _ in b["rounds"] if i <= j)
            if done == len(COURSE):
                state = f"FINISHED {b['t_finish']:5.1f} h"
            elif b["status"] == "aground" and j < k:
                state = "DNF aground"
            elif b["status"] == "time limit" and k >= n - 1:
                state = f"still out {remaining_nm(b, j):4.0f} nm"
            else:
                state = f"{leg_names[done]:<12}{remaining_nm(b, j):4.0f} nm"
            r.set_text(f"{b['name']:<22}\n   {state}   tacks {b['tacks'][j]}")
            r.set_color(b["color"])
        return []

    os.makedirs(OUT_DIR, exist_ok=True)
    anon = boats[0]["name"].startswith("Agent")
    tag = tag + ("_evolving" if seq is not None else "") + ("_anon" if anon else "")
    out = os.path.join(OUT_DIR, f"regatta_{seed}{tag}.gif")
    FuncAnimation(fig, update, frames=frames, interval=1000 / fps).save(out, writer=PillowWriter(fps=fps))
    update(n - 1)
    fig.savefig(out.replace(".gif", ".png"), dpi=120, facecolor=BG)
    plt.close(fig)
    print(f"saved {out}")


def print_results(boats, seed):
    print(f"\nRace #{seed}")
    for pos, b in enumerate(standings(boats), 1):
        res = f"{b['t_finish']:.1f} h" if b["status"] == "finished" else f"DNF ({b['status']})"
        print(f"  {pos}. {b['name']:<24} {res:<22} tacks {b['tacks'][-1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--series", type=int, default=0, help="run N races (seeds 0..N-1) and total points")
    ap.add_argument("--max-hours", type=float, default=45.0)
    ap.add_argument("--wind", choices=("evolving", "static"), default="evolving")
    ap.add_argument("--fleet", choices=("all", "map-vs-blind", "algos"), default="all")
    ap.add_argument("--map-model", default="ppo_regatta_map.zip", help="file in models/sail (map-vs-blind)")
    ap.add_argument("--blind-model", default="ppo_regatta_blind.zip", help="file in models/sail (map-vs-blind)")
    ap.add_argument("--anon", action="store_true", help='label boats "Agent 1", "Agent 2", ... on the figures')
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    fleet = load_fleet(args.fleet, args.map_model, args.blind_model)
    if args.anon:
        print("Label key:")
        for i, f in enumerate(fleet, 1):
            print(f"  Agent {i} = {f['name']}")
            f["name"] = f"Agent {i}"
    def make_wind(seed):
        if args.wind == "evolving":
            return race_wind_seq(seed)
        w, fav = race_wind(seed)
        return w, f"12 kt northerly, more pressure on the {fav}"

    if args.series:
        points = {f["name"]: 0 for f in fleet}
        for seed in range(args.series):
            wind, desc = make_wind(seed)
            boats = race(fleet, wind, args.max_hours)
            print_results(boats, seed)
            for pos, b in enumerate(standings(boats), 1):
                points[b["name"]] += pos if b["status"] == "finished" else len(fleet) + 1
        print("\nSeries (low points win)")
        for pos, (name, pts) in enumerate(sorted(points.items(), key=lambda kv: kv[1]), 1):
            print(f"  {pos}. {name:<24} {pts} pts")
        return

    wind, desc = make_wind(args.seed)
    boats = race(fleet, wind, args.max_hours)
    print_results(boats, args.seed)
    animate(boats, wind, desc, args.seed, fps=args.fps, stride=args.stride,
            tag={"map-vs-blind": "_map_vs_blind", "algos": "_algos"}.get(args.fleet, ""))


if __name__ == "__main__":
    main()
