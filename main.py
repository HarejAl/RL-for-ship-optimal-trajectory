"""
Reinforcement Learning the optimal treajectory for Ship Navigation under Wind Disturbance
==================================================================

This is a scaled-down illustrative example of applying RL to ship navigation 
under a non-time varying wind field.
The script trains and evaluates a TD3 agent to control a ship navigating in a non-time varying wind field.
The wind field is precomputed and loaded from 'WF.pkl'.

Environment:
------------
- Built using a Gymnasium-style custom environment in `env.py`.
- The state space includes: 
    [x, y, vx, vy] - current ship position and velocity
    [xf, yf]       - desired final target position
- The action space includes: 
    [ux, uy] - 2D control force applied to the ship

RL Algorithm:
-------------
- TD3 (Twin Delayed Deep Deterministic Policy Gradient) from `stable-baselines3`
- High action noise is used for better exploration
- The reward encourages proximity to the goal while penalizing time and control effort

Modes:
------
- Set `new_model = True` to initialize a new policy
- Set `training = True` to train and save the model

Visualization:
--------------
- Animated trajectory of the agent navigating the wind field
- Wind intensity and direction are shown with a colormap and quivers
- Control forces are plotted over time

Author: [Your Name]
License: MIT
"""

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from env import CustomEnv

# ========== Setup ==========
device = torch.device('cpu')  # Use 'cuda' for GPU acceleration
plt.close('all')

# ========== Load Wind Field ==========
with open('WF.pkl', 'rb') as f:
    Dict = pickle.load(f)

env = CustomEnv(Dict)

# ========== Model Configuration ==========
new_model = False      # Set to True to start from scratch
training = False       # Set to True to train the model

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=2 * np.ones(n_actions))

if new_model:
    hidden_dim = 128
    num_hidden_layers = 5
    net_structure = [hidden_dim] * num_hidden_layers

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=net_structure, vf=net_structure)
    )

    model = TD3("MlpPolicy", env=env, learning_rate=1e-4, action_noise=action_noise,
                policy_kwargs=policy_kwargs, verbose=1, device=device)
else:
    model = TD3.load("trained_model", env=env, action_noise=action_noise)

# ========== Training ==========
if training:
    total_timesteps = int(1e6)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save("trained_model")

# ========== Evaluation ==========
model = TD3.load("trained_model", env=env)  # Reload model without action noise
obs = env.reset()[0]
## Uncomment if you want determine the initial position
# env.state= np.zeros((4,))
# obs[:4] = 0
tot_rew, time_steps = 0, 0
Done = False
time, x_traj, y_traj, u_x, u_y = [], [], [], [], []

while not Done and time_steps < 500:
    action, _ = model.predict(obs)
    u_x.append(action[0])
    u_y.append(action[1])
    obs, reward, Done, truncated, info = env.step(action)
    x_traj.append(obs[0])
    y_traj.append(obs[1])
    tot_rew += reward
    time.append(time_steps * env.dt)
    time_steps += 1

print(f"Total reward: {tot_rew:.2f}")
print(f"Total time steps: {time_steps}")

# ========== Visualization ==========
fig = plt.figure(figsize=(12, 12))
gs = gridspec.GridSpec(3, 2, width_ratios=[25, 1], height_ratios=[5, 1, 0.2])
ax_traj = fig.add_subplot(gs[0, 0])
ax_cb = fig.add_subplot(gs[0, 1])
ax_u = fig.add_subplot(gs[1, 0])

# --- Wind Field ---
intensity = Dict['Intensity']
direction = Dict['Direction']
xx, yy = Dict['x'], Dict['y']
im = ax_traj.pcolormesh(xx, yy, intensity.T, shading='auto', cmap='viridis')
fig.colorbar(im, cax=ax_cb, label='Wind speed (m/s)')
step = 5
X, Y = np.meshgrid(yy, xx)
u = np.sin(direction)
v = np.cos(direction)
ax_traj.quiver(Y[::step, ::step], X[::step, ::step], -u[::step, ::step], -v[::step, ::step],
               color='white', scale=25, width=0.003, alpha=0.8)

# --- Trajectory ---
ax_traj.plot(x_traj, y_traj, 'cyan', lw=2, label='Optimal Path')
ax_traj.plot(x_traj[0], y_traj[0], 'go', label='Start', markersize=8)
ax_traj.plot(env.XYf[0], env.XYf[1], 's', color='gold', label='Goal', markersize=10)
point, = ax_traj.plot(x_traj[0], y_traj[0], 'o', color='cyan', markersize=10)
ax_traj.set(xlabel='x (-)', ylabel='y (-)', title='Wind Field and Optimal Trajectory')
ax_traj.legend()
ax_traj.grid(True, linestyle='--', alpha=0.5)

# --- Control Actions ---
ax_u.plot(time, u_x, label='Ux', color='blue')
ax_u.plot(time, u_y, label='Uy', color='orange')
point_ux, = ax_u.plot(time[0], u_x[0], 'o', color='blue')
point_uy, = ax_u.plot(time[0], u_y[0], 'o', color='orange')
ax_u.set(xlabel='Time (s)', ylabel='Force (-)', title='Control Actions')
ax_u.legend()
ax_u.grid(True, linestyle='--', alpha=0.5)

# --- Animation Function ---
def update(idx):
    point.set_data([x_traj[idx]], [y_traj[idx]])
    point_ux.set_data([time[idx]], [u_x[idx]])
    point_uy.set_data([time[idx]], [u_y[idx]])
    return point, point_ux, point_uy

ani = FuncAnimation(fig, update, frames=time_steps, interval=25)
plt.tight_layout()
plt.show()

# To save animation as GIF:
# ani.save("wind_trajectory.gif", writer='pillow', fps=10)
