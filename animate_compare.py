"""
Animate DP vs. the DAgger-cloned wind-aware policy on sampled held-out wind fields.

For each of `--n` fields (generated from held-out benchmark seeds < 1e6, unseen in
training), value iteration is solved and the learned policy is rolled out from the same
seeded start/goal. One animated GIF per field shows both ships moving over the wind map,
with a live cost/time readout.

    python animate_compare.py --model models/bc_v2.zip --n 10

Output: output/anim/compare_<i>_seed<seed>.gif
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter

_STROKE = [pe.withStroke(linewidth=3.2, foreground="black")]

from dynamics import ShipParams
from env import ShipEnv
from wind import WindField, generate_wind_field, plot_wind_field
from dp_baseline import ValueIterationPlanner
from benchmark_dp import load_model

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "anim")
LEGACY_FIELD = os.path.join(SCRIPT_DIR, "WF.pkl")


def rollout_policy_with_actions(model, env, start, goal, wind):
    """Roll out an SB3 policy; return traj, actions, J, t, success (like planner.rollout)."""
    base = env.unwrapped
    obs, info = env.reset(options=dict(start=start, goal=goal, wind=wind))
    traj = [base.state.copy()]
    actions = []
    for _ in range(base.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        traj.append(base.state.copy())
        actions.append(np.clip(np.asarray(action, dtype=np.float64), -base.p.u_max, base.p.u_max))
        if terminated or truncated:
            break
    return dict(J=info["J"], t=info["t"], success=info["success"], oob=info["oob"],
                traj=np.array(traj), actions=np.array(actions))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/bc_v2.zip")
    ap.add_argument("--n", type=int, default=10, help="number of fields to animate")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="explicit field seeds (overrides --n)")
    ap.add_argument("--wind", choices=["random", "legacy"], default="random")
    ap.add_argument("--nx", type=int, default=61)
    ap.add_argument("--ny", type=int, default=61)
    ap.add_argument("--nv", type=int, default=13)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2, help="simulate every step, draw every `stride`-th frame")
    ap.add_argument("--tail", type=int, default=40, help="length of the fading trail, in drawn frames")
    ap.add_argument("--device", default=None)
    ap.add_argument("--label", default="DAgger clone", help="legend label for the learned policy")
    ap.add_argument("--with-straight", action="store_true",
                    help="also animate the naive wind-blind straight-line autopilot")
    return ap.parse_args()


def make_animation(wind, start, goal, agents, params, out_path, args):
    """
    agents: list of dicts, each {res, label, color, ls}. `res` has traj, actions, success.
    Draws every agent's ship moving over the wind field with a live cost readout.
    """
    dt = params.dt

    def cum_cost(actions):
        c = dt * (params.time_w + params.ctrl_w * (actions ** 2).sum(axis=1))
        return np.concatenate(([0.0], np.cumsum(c)))

    for a in agents:
        a["traj"] = a["res"]["traj"]
        a["n"] = len(a["traj"])
        a["J"] = cum_cost(a["res"]["actions"])
    n_frames = (max(a["n"] for a in agents) + args.stride - 1) // args.stride

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    im = plot_wind_field(ax, wind, quiver_step=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("wind speed")
    ax.plot(*start, "o", color="white", mec="black", ms=9, zorder=5)
    ax.plot(*goal, "s", color="gold", mec="black", ms=11, zorder=5, label="goal")
    ax.add_patch(plt.Circle(goal, 0.25, fill=False, ec="gold", lw=1.2, ls=":"))

    for a in agents:
        (a["line"],) = ax.plot([], [], color=a["color"], lw=2.6, ls=a.get("ls", "-"),
                               label=a["label"], zorder=6, path_effects=_STROKE)
        (a["dot"],) = ax.plot([], [], "o", color=a["color"], mec="black", mew=1.3, ms=11, zorder=7)
    xmin, xmax, ymin, ymax = wind.extent
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x", ylabel="y")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    txt = ax.text(0.98, 0.02, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
                  family="monospace", bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    def update(f):
        arts = []
        lines = []
        t0 = max(0, f - args.tail) * args.stride
        for a in agents:
            k = min(f * args.stride, a["n"] - 1)
            a["line"].set_data(a["traj"][t0:k + 1, 0], a["traj"][t0:k + 1, 1])
            a["dot"].set_data([a["traj"][k, 0]], [a["traj"][k, 1]])
            done = "goal" if (a["res"]["success"] and k >= a["n"] - 1) else ("t=%.1fs" % (k * dt))
            lines.append(f"{a['label'][:12]:12s} J={a['J'][k]:5.2f}  {done}")
            arts += [a["line"], a["dot"]]
        txt.set_text("\n".join(lines))
        return arts + [txt]

    dp = agents[0]["res"]
    rl = agents[1]["res"]
    gap = ((rl["J"] - dp["J"]) / dp["J"] * 100) if (dp["success"] and rl["success"]) else np.nan
    title = f"{agents[0]['label']} vs {agents[1]['label']}"
    if np.isfinite(gap):
        title += f"   cost gap {gap:+.1f}%"
    ax.set_title(title)
    fig.tight_layout()

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    ani.save(out_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params = ShipParams()
    model, obs_cfg = load_model(args.model)
    seeds = args.seeds if args.seeds is not None else list(range(args.n))

    made = []
    for i, seed in enumerate(seeds):
        wind = WindField.load_legacy(LEGACY_FIELD) if args.wind == "legacy" else generate_wind_field(seed)
        env = ShipEnv(wind, params=params)
        env.reset(seed=seed)
        start, goal = env.state[:2].copy(), env.goal.copy()

        planner = ValueIterationPlanner(wind, goal, params=params, nx=args.nx, ny=args.ny, nv=args.nv,
                                        device=args.device)
        planner.solve(verbose=False)
        dp_res = planner.rollout(env, start)

        rl_env = env
        if obs_cfg is not None:
            from wind_obs import wrap_wind_obs
            rl_env = wrap_wind_obs(env, obs_cfg)
        rl_res = rollout_policy_with_actions(model, rl_env, start, goal, wind)

        agents = [dict(res=dp_res, label="DP", color="cyan", ls="-"),
                  dict(res=rl_res, label=args.label, color="orangered", ls="--")]
        extra = ""
        if args.with_straight:
            from baselines import straight_line_rollout
            st_res = straight_line_rollout(env, start, goal, wind)
            agents.append(dict(res=st_res, label="straight", color="#ffff33", ls=":"))
            extra = f" | straight J={st_res['J']:.2f} ok={st_res['success']}"

        out = os.path.join(OUTPUT_DIR, f"compare_{i:02d}_seed{seed}.gif")
        make_animation(wind, start, goal, agents, params, out, args)
        made.append(out)
        gap = ((rl_res["J"] - dp_res["J"]) / dp_res["J"] * 100) if (dp_res["success"] and rl_res["success"]) else float("nan")
        print(f"[{i:02d}] seed {seed}: DP J={dp_res['J']:.2f} ok={dp_res['success']} | "
              f"RL J={rl_res['J']:.2f} ok={rl_res['success']} gap={gap:+.1f}%{extra}  -> {out}", flush=True)

    print(f"\nsaved {len(made)} animations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
