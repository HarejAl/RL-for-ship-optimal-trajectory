# Plan: a scaled, pure-RL wind-aware routing agent

Branch `feature/pure-rl` (from `feature/ship-scaling`). `main` stays the presentable version.
This file is the working plan: every step lists what it is, how it is checked, and its outcome.
Results go in the log at the bottom, with the hardware and wall-clock time that produced them.

Status keys: `[x]` done · `[~]` in progress · `[ ]` to do

---

## Why this direction

* The DP-teacher clones (`bc_t2`: 83% success, median gap +3.5%) are **data-limited**: every
  new situation (ship type, time-varying wind, real data) needs a new batch of DP solves, and
  the time-dependent DP is the expensive one. Pure RL removes the teacher.
* Earlier pure-RL runs failed for two fixable reasons: a budget of 250k-600k steps, far too
  small for a map input, and a shaped reward that was not the DP cost (speed bias, +34% gap).
* The fix is a **GPU-batched environment** (thousands of ships stepped as one tensor operation)
  and an **exact-cost reward** with potential-based shaping, which provably leaves the optimal
  policy unchanged, so DP stays a fair optimality reference.

## The scaled problem

Dynamics in model units (`dynamics.py`): `v' = u - c_w|v|v - c_a|v-W|(v-W)`, `|u_i| <= u_max`.

| scale | formula | default ship |
| --- | --- | --- |
| calm-water speed V* | sqrt(u_max / c_w) | 4.47 |
| inertia length L* | 1 / c_w | 2.0 |
| response time T* | L* / V* | 0.45 |
| windage ratio kappa | c_a / c_w | 0.5 |

* A ship is characterised by **V\*** and **kappa** (and D/L*, the domain in inertia lengths).
* A faster ship is **exactly** a slower ship in weaker wind: the policy reads `W / V*`
  (tested, including a `bc_t2` rollout). Kappa is not a wind rescaling: it is a policy input.
* Wind maps are stored as a **nondimensional field** `w~ = W / W_ref` with a fixed `W_ref`
  (25 m/s), so `w~` is in [0, 1] for any real weather and keeps absolute severity (a calm
  day is not stretched into a storm, as dividing by each map's own maximum would do).
  A given ship then sees `W / V* = w~ * (W_ref / V*)`.
* A useful single number: **wind authority** `A = kappa * (W_ref / V*)^2`, the wind force on
  a ship held still in a `W_ref` wind relative to its maximum thrust. A < 1: the ship can
  always make headway; A > 1: there are winds it cannot beat head-on.

## Observation design (decision)

Wind is fed as **two Cartesian component maps (W_x, W_y)**, not speed + direction:
the direction angle jumps 359 -> 0 deg and is undefined in calm air, both poison a CNN;
the physics (`v - W`) and bilinear interpolation are linear in the components; and speed is a
function the network gets for free. Optional later: a goal-aligned rotated frame (rotation
equivariance for free data efficiency) and an extra |W|^2 channel (wind force is quadratic).

---

## Steps

### A. Scaling
- [x] **A1** `ShipParams.scales()`, V*-normalised observation, `ReferenceThrustWrapper`,
      `tests/test_scaling.py` (faster ship == weaker wind, exact). Commit `9125673`.
- [x] **A2** Ship catalogue with dimensional properties (container ship, bulk carrier, car
      carrier, motor yacht) -> `(V*, L*, kappa)`; nondimensional wind field; comparison figure:
      same wind map, same straight-to-goal strategy, different ships -> track, speed, thrust,
      hull drag, wind drag, time, energy. Script `ship_scales_demo.py`.

- [~] **A3** Windage by perception instead of a kappa input (Alex's proposal): train on ONE
      kappa_ref and at deployment show the policy `W_eff = sqrt(kappa / kappa_ref) * W`
      (`wind_obs.PerceivedWindWrapper`): a ship more prone to being pushed sees a stronger map.
      Exact for the wind force on a ship at rest; approximate when moving (c_a also drags on the
      ship's own motion). Measured with `kappa_perception_study.py` (DP on the true ship as the
      optimum). Later ablation: kappa-input PPO vs fixed-kappa PPO + perceived wind.

### B. Pure RL on the GPU
- [x] **B1** `gpu_env.py`: batched torch environment. Thousands of ships, each with its own
      generated field (bank refreshed during training, never the held-out seeds), its own kappa
      and wind strength, start and goal. Check: observation identical to
      `WindObsWrapper(add_kappa=True)` and one step identical to `ShipEnv` (unit test).
- [~] **B2** `train_ppo.py`: PPO on the GPU, separate actor and critic CNNs.
      Reward `-stage_cost + gamma*Phi(s') - Phi(s)`, `Phi = -dist / V*`, out-of-bounds penalty.
      Domain randomisation: kappa in [0.05, 0.6], wind multiplier in [0.25, 1.0].
      Smoke run, measure steps/s.
- [ ] **B3** Long run (overnight). Log curves in `output/logs/<tag>/`.
- [ ] **B4** Evaluate against DP and `bc_t2` on the standard held-out cases
      (`benchmark_dp.py --model models/<tag>.pt`): success, optimality gap, by kappa.

### C. Generalisation
- [ ] **C1** Time-varying wind during training (drifting and evolving fields).
- [ ] **C2** Real data: cache an Open-Meteo historical archive for a large region once, train on
      random crops (random box size = random scale, location, start time). Split train/test
      by time period (train on one period, test on a later one).
- [ ] **C3** Inertia regime: randomise D/L*. Real ships have D/L* >> 1 (quasi-steady, Zermelo
      regime) where zooming is exact; our toy domain has D/L* = 5.
- [ ] **C4** Receding-horizon evaluation on real forecasts vs DP re-solved at each update.

### D. Paper
- [ ] **D1** Figures and tables from the log below; ablations (obs design, shaping, kappa input).

---

## Results log

| date | step | what | hardware / time | result |
| --- | --- | --- | --- | --- |
| 2026-10-06 | A1 | scaling tests | CPU, 10 s | 7/7 pass; `bc_t2` on 2x faster ship in 2x wind follows the reference route within 1e-3 |
| 2026-10-06 | A2 | `ship_scales_demo.py`, 100 km box, W_ref 25 m/s, full thrust at the goal | CPU, 5 s | container ship kappa 0.065 A 0.28: +8% time, 1.9 km drift; bulk carrier kappa 0.024 A 0.30: +7%, 1.6 km; car carrier kappa 0.19 A 1.18: +15%, 7.3 km; yacht kappa 0.09 A 1.56: +10%, 10.8 km. Exact collapse (2x thrust, sqrt2 wind): same track, time ratio 1.414. RL toy ship: kappa 0.5, A 2.5, L* 20 km at this scale (D/L* = 6 vs 10-735 for real ships) |
| 2026-10-06 | B1 | GPU env vs CPU env | RTX 4070S, 14 s | 3/3 pass: observations to 1e-5, 40-step trajectories and cost identical |
| 2026-10-06 | B2 | PPO smoke run, 2048 envs x 64 steps, 12 iterations (1.6M steps) | RTX 4070S **shared with another job at 100% GPU and ~11 GB**: ~2-3k steps/s (update 35-60 s/iter, memory spilled to system RAM) | train success 6% -> 86% (curriculum radius 1.0 -> 0.74); held-out success at the final 0.5 radius 6% -> 45%. Learns fast; throughput is entirely limited by the shared GPU |
