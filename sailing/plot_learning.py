"""
Learning curves for the sailing agents: every run scored against the DP optimum during training.

    python sailing/plot_learning.py                     # the algorithm comparison
    python sailing/plot_learning.py --group map         # map vs blind
    python sailing/plot_learning.py --x wall_min        # same curves against wall-clock minutes

Reads the jsonl written by train_rl.py (one line per evaluation: success rate on the 40 held-out
cases, median time gap to DP, manoeuvres per voyage). Those are the comparable curves here --
episode reward is not, because the runs use different reward shaping.
Output: output/sailing/rl/learning_<group>.png
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, REPO_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

LOG_DIR = os.path.join(REPO_DIR, "output", "sailing", "rl")
BG, FG, GRID = "#04121f", "#e8f1f8", "#1e3a4f"

GROUPS = {
    # label -> (log tag, colour)
    "algos": [("PPO", "ppo_time", "#00e5ff"),
              ("A2C", "a2c_time", "#b8f35f"),
              ("DQN", "dqn_time", "#ffd166"),
              ("SAC (continuous)", "sac_time", "#ff8fab"),
              ("PPO, distance shaping", "ppo_regatta_blind", "#c792ea")],
    "map": [("PPO + CNN wind map", "ppo_regatta_map", "#00e5ff"),
            ("PPO, local wind only", "ppo_regatta_blind", "#ff9f1c")],
    "uniform": [("PPO, 36 headings", "ppo_uniform", "#00e5ff"),
                ("SAC, continuous heading", "sac_uniform", "#ff8fab")],
}
PANELS = [("success", "voyages that reach the mark", lambda v: 100 * v, "%.0f%%"),
          ("median_gap_pct", "median time above the DP optimum", lambda v: v, "%.0f%%"),
          ("tacks", "manoeuvres per voyage", lambda v: v, "%.0f")]


def load(tag):
    path = os.path.join(LOG_DIR, f"{tag}_log.jsonl")
    if not os.path.exists(path):
        return None
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in rows[0]} if rows else None


def smooth(y, k=3):
    if len(y) < k:
        return y
    pad = np.concatenate([np.full(k // 2, y[0]), y, np.full(k // 2, y[-1])])
    return np.convolve(pad, np.ones(k) / k, mode="valid")[:len(y)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=tuple(GROUPS), default="algos")
    ap.add_argument("--x", choices=("steps", "wall_min"), default="steps")
    args = ap.parse_args()

    runs = [(label, load(tag), color) for label, tag, color in GROUPS[args.group]]
    runs = [(l, d, c) for l, d, c in runs if d is not None]
    if not runs:
        raise SystemExit("no logs found -- train something first")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), facecolor=BG)
    xlabel = "environment steps (millions)" if args.x == "steps" else "wall-clock minutes"
    for ax, (key, title, fn, fmt) in zip(axes, PANELS):
        ax.set_facecolor(BG)
        for label, d, color in runs:
            x = d[args.x] / (1e6 if args.x == "steps" else 1.0)
            y = fn(d[key])
            ax.plot(x, y, color=color, lw=1.0, alpha=0.30)
            ax.plot(x, smooth(y), color=color, lw=2.4, label=label,
                    path_effects=[pe.Stroke(linewidth=4, foreground=BG, alpha=0.6), pe.Normal()])
        if key == "success":
            ax.axhline(100, color=FG, ls="--", lw=1.2, alpha=0.7)
            ax.text(0.98, 100, " DP: 100%", color=FG, fontsize=9, ha="right", va="bottom",
                    transform=ax.get_yaxis_transform())
            ax.set_ylim(0, 108)
        elif key == "median_gap_pct":
            ax.axhline(0, color=FG, ls="--", lw=1.2, alpha=0.7)
            ax.text(0.98, 0, " DP optimum", color=FG, fontsize=9, ha="right", va="bottom",
                    transform=ax.get_yaxis_transform())
            ax.set_ylim(-5, 60)
        else:
            dp = float(np.nanmean(np.concatenate([d["dp_tacks"] for _, d, _ in runs])))
            ax.axhline(dp, color=FG, ls="--", lw=1.2, alpha=0.7)
            ax.text(0.98, dp, f" DP: {dp:.1f}", color=FG, fontsize=9, ha="right", va="bottom",
                    transform=ax.get_yaxis_transform())
            ax.set_yscale("symlog", linthresh=10)
            ax.set_ylim(0, 300)
        ax.set_title(title, color=FG, fontsize=12, pad=8)
        ax.set_xlabel(xlabel, color=FG, fontsize=10)
        ax.grid(True, color=GRID, lw=0.7, alpha=0.7)
        ax.tick_params(colors=FG, labelsize=9)
        for s in ax.spines.values():
            s.set_color(GRID)

    leg = axes[0].legend(loc="lower right", fontsize=9.5, framealpha=0.25, facecolor=BG, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(FG)
    titles = dict(algos="One boat per RL algorithm - same winds, same reward, same budget",
                  map="Does reading the wind map help?  PPO with and without the CNN map",
                  uniform="Uniform wind: discrete headings (PPO) vs continuous (SAC)")
    fig.suptitle(titles[args.group], color=FG, fontsize=14)
    fig.text(0.5, 0.012, "40 held-out cases, evaluated every 50k steps (thin line) and smoothed (thick). "
             "The time gap counts only voyages that arrive, so it flatters an agent that skips the hard "
             "upwind courses -- read it together with the panel on the left.",
             color=FG, fontsize=8.5, ha="center", alpha=0.85)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    out = os.path.join(LOG_DIR, f"learning_{args.group}{'_wall' if args.x == 'wall_min' else ''}.png")
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
