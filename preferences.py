"""
Cost-weight presets ("preferences") for the multi-objective study.

The stage cost of `dynamics.stage_cost` is

    l(u) = dt * (time_w + ctrl_w * |u|^2)

so a preference is nothing but a pair (time_w, ctrl_w): the exchange rate between
travel time and control energy. Everything else - dynamics, wind, goal disc,
observation - is identical across preferences, so any difference between the
resulting agents comes from the objective alone.

Three presets span a decade of the ratio ctrl_w / time_w:

    fast      ctrl_w = 1e-3   energy is almost free -> run at full thrust
    balanced  ctrl_w = 1e-2   the repo default -> compromise route
    eco       ctrl_w = 1e-1   energy dominates -> slow down and ride the wind

On the held-out DP solutions this spans roughly a factor 2-3 in arrival time and
4-14 in energy, which is a wide enough spread to see three distinct behaviours.

Use `params(name)` to get the `ShipParams` of a preset and `add_pref_arg(parser)`
to give any script the same `--pref` flag.
"""

from dataclasses import replace

from dynamics import ShipParams


# name -> (time_w, ctrl_w, one-line description)
PREFERENCES = {
    "fast":     (1.0, 1e-3, "time priority: energy nearly free, arrive as early as possible"),
    "balanced": (1.0, 1e-2, "trade-off: the repo default weighting"),
    "eco":      (1.0, 1e-1, "energy priority: spend as little thrust as possible"),
}

ORDER = ["fast", "balanced", "eco"]

# consistent colours across every figure of the study (Okabe-Ito, as in journal.mplstyle)
COLORS = {"fast": "#D55E00", "balanced": "#0072B2", "eco": "#009E73"}


def params(name, base=None):
    """ShipParams of preference `name` (dynamics taken from `base`, default ShipParams())."""
    if name not in PREFERENCES:
        raise KeyError(f"unknown preference {name!r}; choose from {sorted(PREFERENCES)}")
    time_w, ctrl_w, _ = PREFERENCES[name]
    return replace(base or ShipParams(), time_w=time_w, ctrl_w=ctrl_w)


def describe(name):
    time_w, ctrl_w, doc = PREFERENCES[name]
    return f"{name} (time_w={time_w:g}, ctrl_w={ctrl_w:g}): {doc}"


def add_pref_arg(parser, default="balanced", dest="pref"):
    """Add a --pref flag listing the presets; `None` means 'use the plain ShipParams default'."""
    parser.add_argument("--pref", dest=dest, choices=list(PREFERENCES), default=default,
                        help="cost-weight preset; " + " | ".join(describe(n) for n in ORDER))
    return parser


def params_from_args(args, base=None):
    """ShipParams from an argparse namespace that may carry --pref, --time-w, --ctrl-w."""
    p = params(getattr(args, "pref"), base) if getattr(args, "pref", None) else (base or ShipParams())
    if getattr(args, "time_w", None) is not None:
        p = replace(p, time_w=float(args.time_w))
    if getattr(args, "ctrl_w", None) is not None:
        p = replace(p, ctrl_w=float(args.ctrl_w))
    return p


def energy(actions, p):
    """Control energy E = sum |u|^2 dt of an action sequence (N, 2) - the cost term
    that `ctrl_w` multiplies, reported unweighted so preferences can be compared."""
    import numpy as np
    a = np.asarray(actions, dtype=np.float64).reshape(-1, 2)
    return float((a ** 2).sum() * p.dt)
