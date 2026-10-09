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

- [x] **A3** Windage by perception instead of a kappa input (Alex's proposal): train on ONE
      kappa_ref and at deployment show the policy `W_eff = sqrt(kappa / kappa_ref) * W`
      (`wind_obs.PerceivedWindWrapper`): a ship more prone to being pushed sees a stronger map.
      Exact for the wind force on a ship at rest; approximate when moving (c_a also drags on the
      ship's own motion). Measured with `kappa_perception_study.py` (DP on the true ship as the
      optimum). **ADOPTED 2026-10-07**: the policy is parameter-free (no kappa input); it is
      trained at kappa_ref = 0.5 with wind strength down to 0.15x so that every real ship
      (kappa >= 0.05 -> perceived factor >= 0.32) interpolates. Later ablation: kappa-input PPO
      (`train_ppo.py --kappa-input 0.05 0.6`) vs this.

### B. Pure RL on the GPU
- [x] **B1** `gpu_env.py`: batched torch environment. Thousands of ships, each with its own
      generated field (bank refreshed during training, never the held-out seeds), its own kappa
      and wind strength, start and goal. Check: observation identical to
      `WindObsWrapper(add_kappa=True)` and one step identical to `ShipEnv` (unit test).
- [x] **B2** `train_ppo.py`: PPO on the GPU, separate actor and critic CNNs.
      Reward `-stage_cost + gamma*Phi(s') - Phi(s)`, `Phi = -dist / V*`, out-of-bounds penalty.
      Parameter-free: fixed kappa_ref = 0.5, perceived wind for other ships (step A3), wind
      multiplier in [0.15, 1.0]. Validation reports success per ship kappa (0.05/0.1/0.25/0.5).
      Smoke run, measure steps/s.
- [x] **B3** Long run (overnight): `ppo_pf_v1`. Log curves in `output/logs/<tag>/`.
- [~] **B4** Evaluate against DP and `bc_t2` on the standard held-out cases
      (`benchmark_dp.py --model models/<tag>.pt`): success, optimality gap, by kappa.

- [ ] **B5** `ppo_pf_v2`: (1) **kappa_ref ~ 0.1** instead of 0.5, inside the real-ship range
      (0.02-0.2): the perceived-wind mapping is exact for the wind push but not for the ship's own
      air drag, and with kappa_ref = 0.5 a real ship is faster than the agent expects for its
      thrust, so it throttles back (see D0: -7% speed, +1.5-2% cost). (2) Wind strength up to
      ~1.5x and weighted toward the top, so sqrt(kappa / kappa_ref) for kappa up to 0.2 and the
      full-strength corner of v1 (87%) are both inside training. (3) Re-test the edge goals.

### C. Generalisation
- [ ] **C1** Time-varying wind during training (drifting and evolving fields).
- [ ] **C2** Real data: cache an Open-Meteo historical archive for a large region once, train on
      random crops (random box size = random scale, location, start time). Split train/test
      by time period (train on one period, test on a later one).
- [ ] **C3** Inertia regime: randomise D/L*. Real ships have D/L* >> 1 (quasi-steady, Zermelo
      regime) where zooming is exact; our toy domain has D/L* = 5.
- [ ] **C4** Receding-horizon evaluation on real forecasts vs DP re-solved at each update.

### D0. Deployment demo on real forecasts
- [x] `real_routes_demo.py`: container ship Lisbon -> Funchal (513 nm) and 20 m yacht Venice ->
      Rovinj (53 nm) on the live Open-Meteo forecast, deployed by scaling only (km per unit,
      V*, sqrt(kappa / kappa_ref)); compared with straight-line sailing at the economical and at
      full throttle. Coastline from the Open-Meteo elevation API (the model has no land).

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
| 2026-10-07 | A3 | `kappa_route_demo.py`: `bc_t2` (kappa_ref 0.5) on ships kappa 0.05-0.6, perceived vs raw wind, 200 held-out cases, no DP | CPU (shared), ~15 min | **Theory confirmed.** Perceived: route deviation grows monotonically with kappa (0.35 -> 0.45 -> 0.80 -> 1.09 -> 1.19); raw stays ~1.0 at every kappa (the agent detours as if it were a kappa 0.5 ship). Low-windage ships gain the most: kappa 0.05 success 100% vs 88%, voyage time 2.24 vs 2.54 (-12%), cost J 3.07 vs 3.59 (-14%). Above kappa_ref (0.6) perceived is slightly worse on success (70% vs 76%): extrapolating beyond the training wind. Optimality vs DP: pending (`kappa_perception_study.py`) |
| 2026-10-07 | A3 | `kappa_perception_study.py`: `bc_t2` vs DP solved on the TRUE ship, 30 held-out cases x 4 kappa | CPU (shared), ~90 min | DP 100% everywhere. Raw wind: success 83-87%, median gap +6.7% (kappa 0.05), +6.1% (0.1), +3.5% (0.25, 0.5). **Perceived wind: 100% / -0.3% (kappa 0.05), 97% / +0.6% (0.1), 93% / +1.2% (0.25), 83% / +3.5% (0.5).** The parameter-free mapping is near-optimal for low-windage (real) ships, better than the agent on its own training ship. Decision: drop the kappa input |
| 2026-10-07/08 | B3 | `ppo_pf_v1`: parameter-free PPO, kappa_ref 0.5, wind x[0.15, 1], 2048 envs x 64 steps, 4 epochs, minibatch 4096 | **RTX 4070 SUPER, 12.0 h, 525.7M steps** (~8.3k steps/s for the first ~7 h while the GPU was shared, 21.7k steps/s alone) | Held-out validation success 98% after 26M steps (~1 h), best 99.2% (iter 3400, 446M steps; per kappa 0.05/0.1/0.25/0.5: 100/100/97.7/99.2%). Policy std 0.5 -> 0.01 |
| 2026-10-08 | B4 | `ppo_pf_v1` vs DP on the true ship (same 30 cases x 4 kappa as `bc_t2`, DP reused) | GPU, ~1 min | **Perceived: 100% / -0.6% (kappa 0.05), 100% / -0.2% (0.1), 97% / +0.9% (0.25), 83% / +4.7% (0.5).** Raw wind: 83-97%, +2.9 to +4.7%. Better than `bc_t2` (100/97/93/83%, -0.3/+0.6/+1.2/+3.5%) except at kappa 0.5 in full wind. No DP teacher used |
| 2026-10-08 | B4 | `ppo_pf_v1`, 1000 benchmark cases per kappa, GPU env, success only | GPU, ~2 min | kappa 0.05: 100% (wind x1.0) / 100% (x0.6); 0.1: 100 / 100%; 0.25: 98.2 / 100%; **0.5: 87.3 / 99.4%**. The only weak corner is the most extreme regime (wind authority A = 2.5, stronger than any real ship in A2) at full wind strength, which was the rare tail of the training distribution. 24% of cases have the goal within 0.5 of the spawn-box edge -> B5 |
| 2026-10-09 | D0 | `real_routes_demo.py`, `ppo_pf_v1`, live Open-Meteo, departures every 6 h over the 3-day window (7 container, 11 yacht) | CPU, ~10 min | Deployment by scaling works: 1 unit = 118.8 km / 12.2 km, wind shown x0.38 / x0.42, decisions every 36.9 / 7.6 min; every voyage arrives. In this week's moderate weather (max 9-15 m/s) the best route is close to the straight line, and the agent sails it: cost J **+1.6 to +2.3%** vs straight at the economical throttle (one yacht departure -0.3%), always ~7% slower and ~10% less energy. Cause found: in calm air the agent cruises at 0.566 of the thrust bound on its own kappa 0.5 ship (optimum 0.577) but at 0.533 on a kappa 0.05 ship (speed 3.14 vs optimal 3.32 units): the real ship is faster than expected for its thrust (lower own air drag) and the agent throttles back. Fix in B5 (kappa_ref ~ 0.1) |
