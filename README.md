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

Multi-field, held-out evaluation (30 generated fields never seen in training, one
start/goal each, DP grid 61x61x13x13). Wind-blind MLP TD3 trained on 200 random fields
(snapshot at ~300k steps, `benchmark_dp.py --n-cases 30 --wind random --model ...`):

| | DP baseline | wind-blind RL |
| --- | --- | --- |
| success rate | 97% | 73% |
| cost gap, 22 cases solved by both | reference | median +34%, mean +67% (10th-90th pct: +20% to +74%) |
| online time per episode | 16 s solve | 43 ms |

Observation-design ablation on the same 200 training fields (TD3, curriculum):

| observation | training success at ~100k steps | comment |
| --- | --- | --- |
| 6-vector, wind-blind | 71% | learns fastest; strong baseline |
| 6-vector + 3x3 wind stencil | 34% | learns, slower |
| flattened 6x6 local crop + 8x8 global map (370-d, MLP) | 20% (16% at 220k) | fails |
| Dict local 16x16 + global 16x16 with CNN extractor (TD3 or SAC) | no successes by 250k | fails |

Conclusion: TD3/SAC from scratch do not extract wind information from map inputs
within the step budgets that fit on a shared GPU; the map channels act as noise for
the critic. The remedy adopted is to use the DP baseline as a teacher ("amortised DP").

### DP as teacher: behaviour cloning of the wind-aware CNN policy

`dp_dataset.py` solved value iteration for 150 training fields x 2 goals (300 solves,
~1 h of GPU), labelling 746k states (DP rollouts from random starts plus random
states) with the greedy DP action. `pretrain_bc.py` regressed the CNN actor on them
(15 epochs, validation on 15 held-out fields: MSE 0.11 in scaled action units).
Same 30 held-out benchmark cases as above, no RL fine-tuning yet:

| | DP baseline | wind-aware clone (`bc_v1`) | wind-blind RL |
| --- | --- | --- | --- |
| success rate | 97% | 63% | 73% |
| cost gap, 18 cases solved by all three | reference | **median +3.3%** | median +31.3% |
| cost gap spread (10th-90th pct, cases solved) | | -0.6% to +15.7% | +20% to +74% |
| online time per episode | 17 s solve | 0.35 s | 0.04 s |

The cloned policy follows the DP routes (see `output/compare_bc_v1.png`), including
detours around vortices and through calm zones, and occasionally beats the coarse-grid
DP. Data matters: the same recipe on 37 fields gave 20-30% success and a 7.6% median
gap. Remaining weakness is the success rate (compounding imitation error: stalling just
outside the goal disc, occasional drift out of the domain), which RL fine-tuning targets.

### Improving the clone: DAgger vs RL fine-tuning

Two ways to raise the success rate were tried on the same 30 held-out cases:

* **DAgger round 1** (`dp_dataset.py --rollout-policy models/bc_v1.zip --rollout-noise 0.5`):
  roll out the clone on 60 training fields x 2 goals, label the 525k visited states
  with the DP action, retrain on the aggregated 1.27M samples -> `bc_v2`.
* **TD3+BC fine-tuning** (`train_wind_aware.py --resume models/bc_v1.zip --demo-data ...
  --bc-weight 1.0 --critic-warmup 5000`, 400k env steps) -> `ft_v1_best`. Plain TD3
  from the clone erodes it within ~10k actor updates (50% -> 15% success) because the
  critic is not yet accurate; the cloning term prevents that.

| policy | success | median gap | mean gap | gap p10 / p90 | time ratio vs DP | exec / episode |
| --- | --- | --- | --- | --- | --- | --- |
| DP baseline | 97% | ref | ref | | 1.00 | 12-17 s solve |
| wind-blind RL (TD3, 600k) | 73% | +34.4% | +66.8% | +20 / +74 | 1.38 | 43 ms |
| clone `bc_v1` | 63% | +3.5% | +7.2% | -0.6 / +15.7 | 1.07 | 347 ms |
| **DAgger clone `bc_v2`** | **67%** | **+1.4%** | **+5.8%** | -1.4 / +19.1 | 1.05 | 219 ms |
| TD3+BC fine-tuned `ft_v1_best` | 67% | +9.7% | +31.3% | +0.3 / +50.2 | 1.15 | 186 ms |

(gaps over the cases solved by both DP and the policy; exec time is dominated by the
600-step timeouts of failed episodes, a successful episode takes ~60 ms on CPU)

DAgger is the better route: it keeps the near-optimal cost and raises success. RL
fine-tuning raises success on the validation set but drifts the policy toward the
reward's speed bias and triples the cost gap, so it is not recommended without a
reward that matches the DP cost exactly (see the shaping note above).

Failure analysis: cases 0, 1, 4 and 9 fail for every learned policy. Three of them have
the goal within 0.5 units of the spawn-box edge (the policy stalls or circles next to
the goal, see `output/compare_bc_v2_hard.png`); case 8 is a DP greedy-rollout failure
(not counted). A DAgger round targeted at near-edge goals is the obvious next step.

---

## Roadmap

1. Done: physically consistent dynamics, seeded environment, wind-field generator, DP baseline and benchmark harness.
2. Done: wind-aware CNN policy via DP-teacher behaviour cloning + one DAgger round (`bc_v2`: 67% success, median gap 1.4% on held-out fields). Next: targeted DAgger rounds for near-edge goals, grid-refinement study of the DP teacher.
3. Preference-conditioned policy: the cost weight ratio as an input, giving the whole fast-to-economical Pareto front from one network.
4. Time-varying wind: receding-horizon execution where the field is swapped at each forecast step, compared with re-solved DP.
5. Real forecast data (for example ERA5 10 m wind) and a 3-DOF ship model.
