import numpy as np
import gym
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
from numpy.typing import ArrayLike, NDArray



class WaterReservoir(gym.Env):
    def __init__(self, water_std: float, water_mean: float = 40.0, 
                 normalizer: Optional[Callable] = None):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        self.hbar = 50.0
        self.rhobar = 50.0
        if normalizer is None:
            self.normalizer = lambda x: x
        else:
            self.normalizer = normalizer
        self.episode_length = None

        self.water_inflow_MEAN = water_mean
        self.water_inflow_STD = water_std
    
    def set_ep_len(self, episode_length: int):
        self.episode_length = episode_length
        return self

    def reset(self):
        self.t = 0
        self.done = False
        self.x = np.random.uniform() * 160    # initial state
        return self.normalizer(deepcopy(self.x))
    
    def epsilon(self):
        return self.water_inflow_MEAN + self.water_inflow_STD * np.random.randn()
    
    def step(self, action: float):
        assert self.episode_length is not None
        assert not self.done
        assert - 1.0 <= action <= 1.0 
        self.t += 1
        if self.t >= self.episode_length:
            self.done = True
        uuu = 0.5 * (action + 1.0)
        u_min = np.max([self.x - 100, 0])
        u_max = self.x
        u_actual = u_min + (u_max - u_min) * uuu
        x_next = self.x + self.epsilon() - u_actual
        rew = np.zeros(2)
        rew[0] = - np.max([x_next - self.hbar, 0])
        rew[1] = - np.max([self.rhobar - u_actual, 0])
        self.x = x_next
        return self.normalizer(deepcopy(self.x)), \
               rew / self.episode_length, \
               deepcopy(self.done), {}


