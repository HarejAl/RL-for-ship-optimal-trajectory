import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.interpolate import RegularGridInterpolator

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, DICT):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(
            low=-10, high=10, shape=(2,), dtype=np.float64
        )

        self.observation_space = spaces.Box(
            low=-11, high=11, shape=(6,), dtype=np.float64
        )
        
        self.state = np.zeros(4,)
        self.state_prev = np.zeros(4,)
        
        self.cd1 = 0.5
        self.cd2 = 0.25
        self.dt = 0.5e-1 #20Hz
        
        self.XYf = np.random.uniform(5,10,2)
        self.state_0 = np.zeros((4,))
        
        self.steps = 0 
        self.rew = 0
        self.x_range = (np.min(DICT['x']), np.max(DICT['x']))
        self.y_range = (np.min(DICT['y']), np.max(DICT['y']))
        
        self.xx = DICT['x']
        self.yy = DICT['y']
        self.intensity = DICT['Intensity']
        self.direction = DICT['Direction']
        
        self.INT_interp = RegularGridInterpolator((self.xx, self.yy), DICT['Intensity'], bounds_error=False, fill_value=10)
        self.DIR_interp = RegularGridInterpolator((self.xx, self.yy), DICT['Direction'], bounds_error=False, fill_value=0)

        self.ctrl_w = 10e-3
        self.time_w = 1e-4
        self.target_w = 1

    @property
    def is_healthy(self):
        x = self.state[0]
        y = self.state[1]

        min_x, max_x = self.x_range
        min_y, max_y = self.y_range

        healthy_x = min_x <= x <= max_x
        healthy_y = min_y <= y <= max_y
        is_healthy = healthy_x and healthy_y

        return is_healthy
    
    def step(self, action):

        x = self.state[0]
        y = self.state[1]
        vx = self.state[2]
        vy = self.state[3]

        ux = action[0]
        uy = action[1]

        w = self.INT_interp([[x, y]])[0]
        d = self.DIR_interp([[x, y]])[0]
        
        Fu_x = (w)**2 * self.cd2 * np.sin(d) + vx*abs(vx) *self.cd2 # Force of the wind
        Fu_y = (w)**2 * self.cd2 * np.cos(d) + vy*abs(vy) *self.cd2
        
        """
        x'' + cx' = u_x - Fux --> because of the convention we used for the direction, the wind force is helping 
        -> x'' = u_x - F_x - cx'
        """
        
        ax = ux - self.cd1 * vx*abs(vx) - Fu_x
        ay = uy - self.cd1 * vy*abs(vy) - Fu_y
        
        vx_next = vx + self.dt * ax
        x_next = x + self.dt * vx
        vy_next = vy + self.dt * ay
        y_next = y + self.dt * vy
        
        next_state = np.array([x_next, y_next, vx_next, vy_next])
        
        self.state = next_state
        reward = self.get_rew(action)
        self.state_prev = self.state
        observation = next_state
        observation = np.concatenate((next_state, self.XYf))
        
        info = {
            "x_position": self.state[0],
        }
        
        terminated = False
        
    
        if abs(x_next - self.XYf[0]) <= 2e-1 and abs(y_next - self.XYf[1]) <= 2e-1:
            terminated = True
            reward += 1e2 # Bonus reward for reaching the goal
        
        truncated = False
        
        if not self.is_healthy:
            terminated = True
         
        return observation, reward, terminated, truncated, info
    
    def get_rew(self, action):
        
        self.steps += 1
        
        time_cost = self.steps*self.time_w
        
        control_cost = np.sum(action**2) * self.ctrl_w
        
        dist2target = np.sqrt((self.state[0] - self.XYf[0])**2 + (self.state[1] - self.XYf[1])**2)
        dist2target_prev = np.sqrt((self.state_prev[0] - self.XYf[0])**2 + (self.state_prev[1] - self.XYf[1])**2)
        target_rew =  -(dist2target - dist2target_prev) * self.target_w
        # target_rew = 1/np.maximum(dist2target, 1e-1) * self.target_w

        rew = - control_cost + target_rew - time_cost # if the agent is going close to the goal, the distance is shortening --> also reward is decreasing --> reason why there is a minus
        
        return rew 
        

    def reset(self, seed=None, options=None):
        
        self.state = np.random.uniform(0, 10, size=(4,))
        self.state[-2:] = 0
        
        self.XYf = np.random.uniform(5,10,2)
        
        observation = np.concatenate((self.state, self.XYf))
        
        self.steps = 0
        
        info = 'reset'
        
        return observation, info

if __name__ == "__main__":
    import pickle

    # Load dictionary
    with open('data.pkl', 'rb') as f:
        Dict = pickle.load(f)
        
    env = CustomEnv(Dict)
    obs = env.reset()[0]
    print(f'xf : {obs[-2]}, yf : {obs[-1]}')
    aa = 10
    a = np.array([aa,aa])
    observation, reward, terminated, truncated, info = env.step(a)
    print(f'x : {observation[0]}, y : {observation[1]}, rew : {reward}')
    a = np.array([aa,aa])
    observation, reward, terminated, truncated, info = env.step(a)
    print(f'x : {observation[0]}, y : {observation[1]}, rew : {reward}')
    a = np.array([aa,aa])
    observation, reward, terminated, truncated, info = env.step(a)
    print(f'x : {observation[0]}, y : {observation[1]}, rew : {reward}')
    a = np.array([aa,aa])
    observation, reward, terminated, truncated, info = env.step(a)
    print(f'x : {observation[0]}, y : {observation[1]}, rew : {reward}')
    a = np.array([aa,aa])
    observation, reward, terminated, truncated, info = env.step(a)
    print(f'x : {observation[0]}, y : {observation[1]}, rew : {reward}')
    a = np.array([aa,aa])
    observation, reward, terminated, truncated, info = env.step(a)
    print(f'x : {observation[0]}, y : {observation[1]}, rew : {reward}')
    


