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
| `preferences.py` | The cost-weight presets of the multi-objective study (`fast` / `balanced` / `eco`) and the helpers every script uses to agree on them. |
| `dp_dataset_prefs.py` | One pass over the training fields producing one DP-teacher dataset per preference, on identical fields, goals, starts and random states. |
| `merge_teacher_npz.py` | Concatenate the shards of parallel `dp_dataset_prefs.py` workers. |
| `compare_preferences.py` | Roll the preference agents out on shared held-out cases; success, time-energy Pareto, cross-cost matrix, figures. |
| `evolving_race.py` | Presentation animation: the three preference agents race across a drifting wind map, with plain-language fuel and time readouts. |
| `race_gallery.py` | Sweep fields x routes for presentable races, store every rollout, score them, render the best and build a contact sheet. |
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

Task settings (`ShipEnv`): an episode ends when the ship enters a disc of radius
`goal_radius = 0.5` around the goal (reaching it exactly is not required); start and goal
are sampled at least `min_start_goal_dist = 4.0` apart so the routing problem is not
trivial. The random wind generator uses a short correlation length (`gust_length` 0.5-1.8)
for choppier fields. Current-task results are in "Results — current task" below; the
"Earlier results" section keeps the original 0.25-disc numbers for reference.

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

## Results — current task (2026-09-10)

Task: 0.5-radius goal disc, start and goal >= 4 units apart, choppier wind
(`gust_length` 0.5-1.8). Wind-aware CNN policy cloned from the DP teacher
(`dp_dataset.py` -> 300 solves, 768k labelled states -> `pretrain_bc.py`).
Held-out benchmark, 30 generated fields unseen in training
(`benchmark_dp.py --n-cases 30 --wind random --model models/bc_t2.zip`):

| policy | success | median gap | mean gap | gap p10 / p90 | online / episode |
| --- | --- | --- | --- | --- | --- |
| DP baseline | 100% | ref | ref | | 4 s solve |
| **clone `bc_t2`** (behaviour cloning only) | **83%** | **+3.5%** | +9.5% | +0.8 / +27 | 0.13 s |
| DAgger clone `bc_t2_dag` | 77% | +3.1% | +10.5% | +0.8 / +35 | 0.13 s |

Findings on the updated task:
- Widening the goal disc from 0.25 to 0.5 raised the clone's success from 63% to **83%**
  at the same ~3.5% median cost gap — the old near-miss stalls now count as arrivals.
- The **DAgger round no longer helps** (77% < 83%; its failures are a strict superset of
  the clone's). DAgger previously fixed near-goal stalls, which the wider disc removed, so
  plain behaviour cloning is now the recommended recipe. `bc_t2` is the best model.
- The 5 remaining `bc_t2` failures are genuinely hard: two goals sit within 0.3 units of
  the domain edge (the policy overshoots and exits), three are long routes through choppy
  wind (DP cost 4.4-7.1). Targeted data near the edge and stronger-wind fields is the next
  lever, not DAgger.
- DP now solves all 30 cases (was 97%) and converges in ~4 s (choppier fields need fewer
  value-iteration sweeps). NB: value iteration is the *optimality* reference, not a fast
  online planner — for the compute-saving claim, benchmark against isochrone/Dijkstra
  weather routing or a fast-marching HJB solver as the speed reference.

---

## Multi-objective agents: time priority vs energy priority (2026-09-22)

The stage cost is `l(u) = dt * (time_w + ctrl_w |u|^2)`, so an agent's *preference* is
nothing but the exchange rate between travel time and control energy. Three presets
(`preferences.py`) span a decade of that ratio:

| agent | `ctrl_w` | intent |
| --- | --- | --- |
| `fast` | 1e-3 | energy nearly free: arrive as early as possible |
| `balanced` | 1e-2 | the repo default weighting |
| `eco` | 1e-1 | energy dominates: spend as little thrust as possible |

Everything else — dynamics, wind generator, goal disc, observation, network — is identical,
so any difference between the agents comes from the objective alone.

### Pipeline

Training from scratch is not an option here (the CNN observation does not learn under plain
TD3, see the ablation below), so each agent is a behaviour clone of a DP teacher solved under
*its own* cost:

```bash
python dp_dataset_prefs.py --n-fields 75 --goals-per-field 1 --rollouts 20 --random-states 2500
python merge_teacher_npz.py --inputs dp_teacher_pref dp_teacher_prefB --out dp_teacher_all
python pretrain_bc.py --data data/dp_teacher_all_fast.npz --tag pref_fast --pref fast --epochs 15
python compare_preferences.py --n-cases 200 --tag-prefix pref
```

`dp_dataset_prefs.py` draws each case once and labels it with all three teachers: same fields,
same goals, same rollout starts, same random states. `ValueIterationPlanner.set_cost_weights`
re-targets the planner at a new cost while keeping the successor-cell and interpolation-weight
precompute, which depends on the dynamics only — so the second and third teacher of a case cost
one value iteration each and nothing else.

Two facts make the evaluation cheap. First, a trajectory's cost under *any* preference is an
affine function of two numbers,

    J_p = time_w(p) * T + ctrl_w(p) * E,    T = arrival time,  E = sum |u|^2 dt,

so one rollout per agent can be re-priced under every objective. Second, the DP optimum for
each objective on a benchmark case comes from one shared precompute plus three value iterations.

### Result: the three agents are genuinely different policies

200 held-out generated fields, unseen in training; the cross-cost matrix is the mean cost over
the 24 cases all three agents solved.

| agent | success | median `T` | median `E` | mean `T` | mean `E` | online |
| --- | --- | --- | --- | --- | --- | --- |
| `fast` | 36% | 2.25 | 271 | 2.09 | 270 | 43 ms |
| `balanced` | 36% | 2.95 | 143 | 2.65 | 156 | 56 ms |
| `eco` | 29% | 4.50 | 41 | 4.87 | 90 | 83 ms |

Cross-cost matrix (mean `J`, best in each column in bold):

| agent \ objective | `J_fast` | `J_balanced` | `J_eco` |
| --- | --- | --- | --- |
| `fast` | **2.36** | 4.79 | 29.14 |
| `balanced` | 2.80 | **4.21** | 18.24 |
| `eco` | 4.96 | 5.77 | **13.90** |

Every objective is won by the agent trained for it — the diagonal is minimal in all three
columns, which is the property that makes these three distinct agents rather than three noisy
copies of one policy. The behavioural span is 2.3x in arrival time against 3.0x in energy.
`output/preference_trajectories.png` shows why: `fast` and `balanced` stay near the direct
line at high thrust, while `eco` takes visibly longer detours that ride the wind and arrives
with a fraction of the energy.

### Caveat: success rate is data-limited, not method-limited

These clones reach 29-36% success, against the 83% of `bc_t2` on the same task. The difference
is the teacher dataset: `bc_t2` was cloned from 300 (field, goal) pairs and 768k labelled
states, these agents from 74 pairs and 250-310k states (the data run was stopped at half its
planned length). This matches the trend already recorded for the original task — 37 fields gave
20-30% success, 150 fields x 2 goals gave 83% — so the fix is more DP solves, not a different
recipe. Two things that do *not* help were checked on the `balanced` agent: rebalancing the mix
towards on-distribution rollout states (`--rollout-frac 0.5`) cut validation MSE from 0.22 to
0.13 but left success at 23%, and keeping the best-validation epoch (`--keep-best`) changed
nothing. The relative comparison between the three agents is unaffected — they are trained,
evaluated and scored on identical data budgets and identical cases.

### Showing it to a non-technical audience

`evolving_race.py` races the three agents across a wind map that drifts while they sail, with
the jargon stripped out: the agents are labelled IN A HURRY / BALANCED / FUEL SAVER, the
readouts are hours at sea and a fuel bar scaled to the thirstiest agent, and the wind scale runs
the wind banded into six
plain shades of grey-blue.

```bash
python evolving_race.py
```

On the default case (generated field 31 drifting 6 units east and 3 north, corner to corner,
10.8 of the 12 map units apart) the three agents tell the story without a caption: the hurried
one arrives in 11.6 h having burned all of its fuel budget, the balanced one 4 h later on
half, and the fuel saver dives south, picks up a favourable flank and arrives at 24.3 h on a
quarter. The routes are visibly different, which is the point - it is not three speeds along
one line.

Five demos, one script. Each writes three things: `_figure.png` (one map, the finished routes,
the numbers - the still a slide or a paper wants), `_panels.png` (four stages across the voyage)
and a `.gif`.

```bash
python evolving_race.py --scenario fixed --field-seed 31 --prefs balanced --tag _solo
python evolving_race.py --scenario fixed --field-seed 31
python evolving_race.py --scenario drift --field-seed 31 --prefs balanced --tag _solo
python evolving_race.py --scenario drift --field-seed 31
python evolving_race.py --scenario real --region bay_of_biscay --start 9.0 8.4 --goal 0.8 1.4
```

(the four above take `--start 0.8 1.4 --goal 9.0 8.4 --max-steps 450`)

| output | map | agents | result |
| --- | --- | --- | --- |
| `race_fixed_solo` | one generated field, unchanging | one | the base problem: 24.0 h, 306 fuel |
| `race_fixed` | the same field | three | 10.7 / 14.3 / 24.0 h, fuel 100 / 54 / 22% |
| `race_drift_solo` | the same field, drifting | one | the map moves under a single route |
| `race_drift` | the same field, drifting | three | 11.6 / 15.7 / 24.3 h, fuel 100 / 49 / 24% |
| `race_real` | Open-Meteo, Bay of Biscay | three | 17.9 / 32.3 / 40.9 h, fuel 100 / 37 / **10%** |

All five share field 31 (bar the forecast) and the same corner-to-corner route on purpose, so
they can be shown in sequence: one agent on a map that holds still, then the same agent when the
map will not, then what three different instructions do to it, then the same thing on a real
forecast. A lone agent is drawn in the sailing demos' own agent cyan (`#00e5ff`).

`real` pulls hourly Open-Meteo 10 m wind (no API key, cached under `output/cache`) and runs it on
**the forecast's own clock**: the field metadata gives 62 km per model length unit and 2.5 (m/s)
per model speed unit, so one model time unit is 6.9 real hours and the weather advances exactly
as fast as the forecast says. The 500 km crossing takes the hurried agent 17.9 h and the thrifty
one 40.9 h on a tenth of the fuel. Two caveats: real forecast fields are much weaker than the
generated training fields (this window peaks at 12.9 m/s against the generator's 25), so the
colour scale is auto-fitted to each window or the map renders blank; and on most real routes the
`fast` agent simply times out - of 15 region-route combinations only three had all three agents
arriving.

The animations play at `--fps` x `--stride` simulation steps per second: 12 x 1 by default, so
every step is drawn and a crossing takes 11-17 seconds. (It used to be 20 x 2, which was three
times faster and skipped every other step.) `--hold-s` keeps the finished picture on screen at
the end.

Two looks. `--style sailing` (the default) is the house style of this repo, identical to the
sailing demos: the windy.com speed palette on the same dark ground (`#04121f`), white quivers,
and a drifting particle flow in the animations, so a talk can cut between the ship and the
sailboat work without the audience re-learning the picture. The clutter is still gone - no
lettering on the map, no agent subtitles - the colour is the wind and nothing else.
`--style simple` is the ink-saving alternative: one hue banded into six flat steps on white.

Both styles band the speed into flat steps rather than a smooth ramp - six for `simple`, sixteen
for `sailing`, which still reads as a continuous weather map. That is a file-size decision as
much as a visual one. A smooth gradient dithers into hundreds of near-identical shades, and with
every step drawn the wind map is repainted constantly: the colour animations came out at 17 MB
banded at 16 steps they are 3-5 MB, and the light style drops from 14 MB to about 2. Requantising
to a shared palette afterwards helps the banded maps (`--gif-colors`, on by default) but ruined
the smooth ones - at 64 colours the background crowded out the tracks and two agents came out the
same colour.

The weather is a generated field sliding across the domain rather than one of the synthetic set
pieces in `evolving_scenario_demo.py`. Those are deliberately stronger than anything the wind
generator produces (cyclones at strength 11, deepening gale fronts), which makes them good
stress tests and bad demos: the clones are far outside their training distribution there and
simply thrash - measured, all three time out or wander on every set piece tried. Drifting a
generated field keeps magnitudes and correlation lengths exactly those of training, so the
weather moves without leaving the distribution.

Because these clones only arrive on about a third of crossings, a presentable race has to be
found rather than assumed. `race_gallery.py` does the finding:

```bash
python race_gallery.py --seeds 1 80 --top 8
```

It races the three agents through every combination of generated field and named route, scores
each case on what makes the animation worth watching - all three arrive, in the expected order,
by visibly different paths, across lively weather, with a big fuel ratio - then renders the best
and lays them out in `output/race_gallery_contact.png` to choose from. The sweep runs at about
2 s per case on CPU; 400 cases take a quarter of an hour.

Measured over 80 fields x 5 routes: **76 of 400 cases have all three agents arriving and 39 are
fully ordered**, which is the success rate of the clones showing through. Say so if anyone asks
what a typical crossing looks like.

EVERY case is stored in `output/race_runs/<name>.npz` - trajectories, thrusts, fuel curves,
outcomes, and the case configuration - whether it scored well or not, and indexed with its
metrics in `output/race_gallery.csv`. Figures can therefore be remade without re-simulating:

```python
from evolving_race import load_runs, drifting_weather
runs, prefs, meta = load_runs("output/race_runs/s48_ne.npz")
field = drifting_weather(1.0, seed=meta["field_seed"], drift=tuple(meta["drift"]))
```

That is how the contact sheet is drawn, and it is the cheap way to try a different visual
treatment on cases that are already known to work.

Still to run (needs the GPU): the remaining 76 fields of the teacher datasets, and
`compare_preferences.py` without `--no-dp`, which adds the DP optimum for each objective and
therefore the per-objective optimality gap of each agent.

---

## Earlier results — original task, 0.25 disc (2026-09-08)

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
2. Done: wind-aware CNN policy via DP-teacher behaviour cloning (`bc_t2`: 83% success, median gap 3.5% on the current task; DAgger no longer needed with the 0.5 disc). Next: harder-case data (edge goals, stronger wind), a fast competitive planner as the speed reference, grid-refinement study of the DP teacher.
3. Partly done: **separate** agents per cost weighting (`pref_fast`, `pref_balanced`, `pref_eco`), each behaviour-cloned from its own DP teacher — each wins its own objective, spanning 2.3x in arrival time against 3.0x in energy (see the multi-objective section). Outstanding: finish their teacher datasets (74 of 150 fields collected, which is what caps success at ~30%) and add the DP optimality gap per objective. Then the **preference-conditioned** policy: the cost-weight ratio as a network input, giving the whole fast-to-economical Pareto front from one network, with these three agents as the baseline to beat.
4. Time-varying wind: receding-horizon execution where the field is swapped at each forecast step, compared with re-solved DP.
5. Real forecast data (for example ERA5 10 m wind) and a 3-DOF ship model.
