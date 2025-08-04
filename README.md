RL Ship Navigation in Wind Fields (Scaled Example)

This repository provides a minimal example of applying Reinforcement Learning to a ship navigation problem under static wind disturbance. The goal is for an agent to learn how to steer a ship from a random initial position to an arbitrary destination by minimizing travel time and control effort.

The environment is implemented using Gymnasium, and the agent is trained using the TD3 algorithm from `stable-baselines3`.

---
Project Overview

- The wind field is precomputed and loaded from `WF.pkl`.
- The agent observes:
  - Current position and velocity `[x, y, vx, vy]`
  - Desired destination `[xf, yf]`
- The agent outputs:
  - Control actions `[ux, uy]` (forces)
- The reward function balances:
  - Reducing the distance to the target
  - Penalizing control energy and time steps

---
Current Features

- TD3 training on a fixed wind field
- Realistic reward shaping with tunable weights
- Wind direction visualization with intensity map
- Animated trajectory showing path and control actions

---
Ongoing Work

The current version uses a static wind field, but upcoming improvements include:

- Wind-aware learning: feeding the wind field (or a compressed representation) to the agent as part of the observation space. This would allow the agent to generalize across different wind maps.
- Multi-agent formulation: training multiple agents with varying weight preferences on control effort vs. travel time. This enables trajectory personalization:
  - "Faster but less efficient"
  - "Slower but more economical"
  - Users could choose the agent behavior that best suits their operational needs.

---
