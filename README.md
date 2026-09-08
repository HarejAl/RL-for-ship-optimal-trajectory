# RL Ship Navigation in Wind Fields

Reinforcement learning for minimum-cost ship routing through a 2D wind field, with an
exact dynamic-programming baseline so that learned policies can be scored on their
optimality gap and on the computational effort they save.

The long-term goal is a wind-aware agent that reads the wind map (direction and speed)
and can therefore be applied to unseen and time-varying forecasts in a receding-horizon
loop, amortising the cost of re-solving the routing problem each time the forecast updates.

---

## Repository layout

| File | Purpose |
| --- | --- |
| `dynamics.py` | Ship model and stage cost shared by the environment and the baseline. Works with numpy and torch. |
| `wind.py` | `WindField` container (bilinear interpolation of the velocity components), legacy `WF.pkl` conversion, random wind-field generator, plotting helper. |
| `env.py` | Gymnasium environment `ShipEnv`. Seeded resets, start and goal sampled in the domain, cost `J` tracked in `info`. |
| `dp_baseline.py` | `ValueIterationPlanner`: semi-Lagrangian value iteration on a 4D `(x, y, vx, vy)` grid, torch/GPU vectorised, greedy policy rollout in the environment. |
| `run_dp_baseline.py` | Solve one case, print cost, time and solve statistics, save a figure to `output/`. |
| `benchmark_dp.py` | Solve many cases; optionally roll out a stable-baselines3 model on the same cases and report the optimality gap. |
| `wind_obs.py` | Wind-aware observations: `WindObsWrapper` (state vector + ego-centric wind crop + coarse global map), `WindFieldPool`, `WindCNNExtractor` for SB3. |
| `train_wind_aware.py` | Train TD3 or SAC with the CNN extractor on a pool of generated fields, validating on held-out fields. |
| `compare_policy.py` | Plot DP and RL trajectories side by side on held-out cases. |
| `tests/test_core.py` | Unit and sanity tests (interpolation, dynamics, environment, DP on zero wind and uniform wind). |
| `legacy/` | The original single-field TD3 code (`env.py`, `main.py`). `trained_model.zip` was trained with this legacy environment and is **not** compatible with the new dynamics. |
| `WF.pkl` | The original precomputed wind field (speed and direction). |

---

## Model

Point-mass ship with unit mass, thrust `u` bounded per axis, quadratic hull drag through
the water (at rest) and quadratic wind drag on the velocity relative to the air:

```
v_dot = u - c_w |v| v - c_a |v - W| (v - W)
x_dot = v
```

Integration is semi-implicit Euler at 20 Hz. The objective minimised by both the DP
baseline and the RL agent is

```
J = sum_k dt * (time_w + ctrl_w * |u_k|^2)
```

that is, travel time plus control energy with tunable weights (`ShipParams`). The RL
reward is `-stage_cost` plus a potential-based shaping term on the distance to the goal
and terminal bonuses, so the ranking of successful trajectories is unchanged. Sweeping
`ctrl_w` against `time_w` produces the "fast" versus "economical" trade-off for both methods.

Note on scales: with the default coefficients a 10 unit wind exerts a force of 25 while
the maximum thrust is about 14, so regions of very strong headwind are physically
unreachable. The value function shows them as saturated cost.

---

## DP baseline

`ValueIterationPlanner` discretises the state on a regular grid, evaluates the same
`ship_step` as the environment for every (state, action) pair once, stores the
successor cell and the 16 multilinear interpolation weights, and iterates the Bellman
equation until the value change is below tolerance. The policy is one-step lookahead on
the value function with a finer action grid, executed in the environment, so the
baseline and the RL agent are compared on identical dynamics and cost.

The solve must be repeated for every (wind field, goal) pair. On an RTX 4070 the default
grid (61x61x13x13 states, 25 actions) converges in roughly 500 iterations and 10 s. That
wall-clock time is the quantity an amortised learned policy is compared against.

---

## Wind-aware policy

The policy observes a Dict:

| Key | Shape | Content |
| --- | --- | --- |
| `vec` | (6,) | goal offset, velocity, absolute position, all scaled to O(1) |
| `local` | (3, 16, 16) | ego-centric crop of half-width 2 units: `wx`, `wy`, inside-domain mask |
| `global` | (4, 16, 16) | whole field downsampled: `wx`, `wy`, gaussian blob at the ship, blob at the goal |

A small CNN per map plus an MLP for the vector feed a TD3 (or SAC) actor-critic. Training
samples a new field from a pool of 200 generated fields at every reset. Seed namespaces are
disjoint: training fields use seeds from 1,000,000, validation from 2,000,000, and the
benchmark cases in `benchmark_dp.py` use seeds below 10,000, so evaluation is always on
unseen fields.

---

## Usage

```bash
pip install -r requirements.txt
python -m pytest tests -q
python run_dp_baseline.py                           # legacy field, seeded start/goal
python run_dp_baseline.py --wind random --seed 7    # generated field
python benchmark_dp.py --n-cases 20 --wind random   # many cases, CSV in output/
python benchmark_dp.py --n-cases 20 --model path/to/model.zip   # add RL optimality gap
python train_wind_aware.py --timesteps 1000000 --n-envs 8 --gradient-steps 4 --tag td3_v1
python benchmark_dp.py --n-cases 50 --model models/td3_v1_best/best_model.zip
python compare_policy.py --model models/td3_v1_best/best_model.zip --seeds 0 1 2 3
```

Trained models go to `models/` (git-ignored) with a `.json` sidecar describing the
observation wrapper, and logs to `output/logs/<tag>/`.

The legacy demo still runs with `python legacy/main.py`.

---

## Preliminary results (2026-09-08)

Single-field control experiment: a plain MLP TD3 policy trained on the legacy field
(300k steps, goal-radius curriculum), benchmarked on 20 seeded start/goal pairs of the
same field against the DP baseline (61x61x13x13 grid). Numbers from
`benchmark_dp.py --wind legacy --model models/td3_plain_legacy_best/best_model.zip`:

| | DP baseline | RL policy |
| --- | --- | --- |
| success rate | 95% (19/20) | 80% (16/20) |
| cost J, cases solved by both | reference | median gap +28%, mean +49% |
| online time per episode | 35 s solve (shared GPU; ~10 s idle) | 27 ms |

Observations:
- The RL policy is usually *faster* than DP but spends more thrust, so its cost is
  higher: the distance-shaping term plus discounting bias it toward speed. Reducing the
  shaping weight late in training (or using the exact `gamma*Phi(s') - Phi(s)` form) is
  the obvious next fix for the gap.
- The one DP failure is a greedy-lookahead limit cycle near a goal in an 8 m/s
  crosswind; a finer velocity grid (`--nv 21`) solves it. The value function
  overestimates the true cost on such cases, so the reported gaps are conservative for
  the RL side. A grid-refinement study is required before publication.
- Wind-aware CNN policies on 200 random fields are still training; the wind-blind
  multi-field MLP is the ablation they must beat.

---

## Roadmap

1. Done: physically consistent dynamics, seeded environment, wind-field generator, DP baseline and benchmark harness.
2. Done (code): wind-aware policy with a CNN over the local crop and the global map, trained on random fields, evaluated on held-out fields against the DP baseline. Training runs and results in progress.
3. Preference-conditioned policy: the cost weight ratio as an input, giving the whole fast-to-economical Pareto front from one network.
4. Time-varying wind: receding-horizon execution where the field is swapped at each forecast step, compared with re-solved DP.
5. Real forecast data (for example ERA5 10 m wind) and a 3-DOF ship model.
