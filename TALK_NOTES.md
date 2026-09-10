# Talk notes — learning to route a ship through wind

Everything here is reproducible from this repo. Figures are in `output/`; the numbers below
are the ones actually produced by the scripts named beside them. Caveats are listed
explicitly at the end — several of them are the kind of thing an audience will probe.

---

## The one-line story

Optimal weather routing is a solved problem *in principle* (dynamic programming) but it has
to be re-solved from scratch for every wind map and every destination. If a network learns
to **read the wind map**, routing becomes a forward pass: milliseconds instead of seconds,
and it keeps working when the weather moves.

---

## Act 1 — routing actually matters

**Figure:** `output/showcase/poster.png` (five panels, one slide) and the individual
`output/showcase/{storm,jet,barrier,twin,front}.png` + `.gif`.

Five designed wind fields where the optimal route is visibly non-trivial. The number on
each panel is the maximum lateral detour off the straight line:

| Scenario | Detour | What the audience sees |
| --- | --- | --- |
| Storm on the direct route | 24% | Arcs around the cyclone rather than through it |
| Favourable jet north of the rhumb line | 35% | Climbs into the fast lane, rides it, drops to the goal |
| Headwind barrier with one gap | 37% | Threads the single gap in the wall |
| Counter-rotating pair blocking the lane | 59% | Swings right around the blocking pair |
| Shear front, gale above / calm below | 60% | Dives south of the gale, runs across, climbs back |

These detours are *forced by physics*, not drawn for effect: wind force is `0.25*w^2` and the
thrust limit is ~14, so anything above ~7.5 wind units cannot be beaten head-on. The blocking
features are built above that threshold; the favourable jet is built below it.

> Be upfront: **these five fields are designed for exposition.** Real forecasts are much
> smoother, and on them the optimum is close to a straight line. Every *quantitative* claim
> below comes from real forecast data instead.

---

## Act 2 — the obvious approach fails (worth telling)

Plain reinforcement learning (TD3 and SAC) with a CNN reading the wind map **never learned**
within the budget available. Observation-design ablation, all on the same 200 training fields:

| Observation the agent sees | Training success @ ~100k steps |
| --- | --- |
| 6-vector, wind-blind | 71% |
| 6-vector + 3x3 wind stencil | 34% |
| Flattened local crop + global map (MLP) | 20% (16% at 220k) |
| Local 16x16 + global 16x16 via CNN (TD3 or SAC) | no successes by 250k |

The pattern is monotone and uncomfortable: **the more map you feed it, the worse it learns.**
The map channels behave as noise for the critic, and the wind-blind baseline learns fastest.
This is why the project pivoted.

---

## Act 3 — use dynamic programming as a teacher

The key asymmetry: one value-iteration solve produces the optimal action at **every state**
of that (wind field, destination) — not one trajectory, but a whole feedback policy. So a
single solve is worth thousands of labelled training examples.

Pipeline (`dp_dataset.py` → `pretrain_bc.py` → `benchmark_dp.py`):
300 solves → **768k labelled states** → behaviour cloning of the CNN policy.

Held-out benchmark, 30 generated fields never seen in training:

| | DP baseline | Learned policy (`bc_t2`) |
| --- | --- | --- |
| Arrival rate | 100% | **83%** |
| Median cost above optimum | reference | **+3.5%** |
| Online cost per voyage | ~4 s solve | **0.13 s** |

A DAgger round was tried and **did not help** (77% vs 83%; its failures were a strict superset).
It had helped on an earlier, stricter task, where it fixed ships stalling just outside a small
goal disc; widening the arrival circle removed that failure mode and made DAgger redundant.
Plain behaviour cloning is the recommended recipe.

---

## Act 4 — it transfers to real weather, zero-shot

**Figure:** `output/staleness_study.png`, plus `output/real_wind_ligurian.png` for a single
real field with the DP solution on it.

Real 10 m wind from Open-Meteo (free, no API key) via `WindField.from_openmeteo`. The policy
had **only ever seen synthetic random fields.** Across 36 cases (3 Atlantic regions x 4
departure times x 3 routes, voyages of 17-49 h):

- **Arrival rate 100%** for the optimal plan, the stale plan and the learned policy.
- **Learned policy: +1.4% median above the DP optimum** (mean +2.7%, p90 +13%).

That is the result I would lead with. A network trained on procedurally generated noise
routes a ship through real Atlantic weather within a couple of percent of optimal.

---

## Act 5 — the weather moves while you sail

**Figure:** `output/receding_horizon.gif` and `output/receding_horizon_summary.png`.

The physical scales make this honest: at ~89 km per model unit and 2.5 (m/s) per wind unit,
one Atlantic crossing takes **20-35 real hours**, so the forecast genuinely evolves under way.

Cost of planning once at departure and never looking again:

| | Median | Mean | p90 | Worst |
| --- | --- | --- | --- | --- |
| Stale forecast penalty | +1.6% | +3.6% | +22.1% | **+48.8%** |

The honest reading, and the interesting one: **most of the time it barely matters, and
occasionally it costs you half again as much** — and you cannot tell at departure which kind
of day you are having. That is the argument for a policy that re-reads the map continuously,
given that adapting costs ~1 ms and re-solving costs seconds.

Note the distribution is two-sided: sometimes the weather *improves* and the voyage comes in
cheaper than the departure forecast promised (down to -35%).

---

## Caveats — say these before someone else does

1. **DP is the optimality reference, not the speed reference.** Value iteration on a 4D grid
   is a textbook method, not state of the art for trajectory optimisation. A tuned
   isochrone / Dijkstra weather router, a direct-collocation NLP, or a fast-marching HJB
   solver would produce one route far faster than my ~4 s solve. The defensible claim is
   *amortisation* — a whole feedback policy for free at run time — not "we beat the best planner".
   A fair fast baseline is the top item of future work.
2. **The ship model is a 2D point mass** with quadratic hull and wind drag, nondimensional.
   Not a 3-DOF vessel with a real power curve or seakeeping limits.
3. **The five showcase fields are designed**, not sampled from weather. They illustrate; they
   do not evidence.
4. **No grid-refinement study yet** for the DP teacher. One earlier failure was traced to a
   coarse velocity grid (`nv=13`) causing a limit cycle near the goal, fixed at `nv=21`.
   Reviewers will ask for convergence evidence.
5. **Remaining failures are structured**: of 30 held-out cases, 5 fail — two with the goal
   within 0.3 units of the domain edge (the ship overshoots and exits), three long routes in
   choppy wind. The fix is targeted training data, not more DAgger.
6. **Lat/lon boxes are stretched** to a square nondimensional domain (each axis scaled
   independently), so distances are nondimensional. `field.meta` carries `km_per_unit` and
   `ms_per_unit` to recover physical units.

---

## A methodological point worth one slide

Two results in this project were nearly reported wrongly, and a control caught each:

- A dramatic "the fixed plan gets blown off course by the mistral, the adaptive policy
  survives" case. Running the same plan under the weather it *assumed* showed it failed
  identically — the plan was never viable. Root cause was a scaling flaw that made the ship
  unable to make headway against an 11 m/s breeze.
- RL fine-tuning on top of the clone looked like a success on the validation set (success
  rate up) while actually **tripling** the cost gap, because it optimises a shaped reward
  that favours speed over the true cost.

Both are good "how we work" material, and both are why the numbers above have controls
attached to them.

---

## Reproducing the figures

```bash
python showcase.py --animate            # Act 1 poster + scenario stills/GIFs
python benchmark_dp.py --n-cases 30 --wind random --model models/bc_t2.zip   # Act 3
python staleness_study.py --model models/bc_t2.zip                            # Acts 4-5
python receding_horizon_demo.py --model models/bc_t2.zip --lat 48 56 --lon -25 -12 \
       --hours 44 --ref-speed 25                                              # Act 5 animation
```

Best model: `models/bc_t2.zip` (behaviour cloning from the DP teacher).
