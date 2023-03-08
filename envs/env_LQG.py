import numpy as np
import gym
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
from numpy.typing import ArrayLike, NDArray




class LQG(gym.Env):
    def __init__(self, a_max: float, n_obj: int, 
                 noise_std: float, normalizer: Optional[Callable] = None):
        super().__init__()
        assert a_max > 0.0 and n_obj >= 1
        assert noise_std >= 0
        self.a_max = a_max
        self.n_obj = n_obj
        self.noise_std = noise_std
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_obj,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_obj,))
        self.xi = 0.1
        if normalizer is None:
            self.normalizer = lambda x: x
        else:
            self.normalizer = normalizer
        self.max_episode_length = None
    
    def set_ep_len(self, max_episode_length: int):
        self.max_episode_length = max_episode_length
        return self

    def reset(self):
        self.t = 0
        self.done = False
        self.x = np.array([10.0] * self.n_obj)    # initial state
        return self.normalizer(deepcopy(self.x))
    
    def step(self, action: NDArray):
        assert self.max_episode_length is not None
        assert not self.done
        self.t += 1
        if self.t >= self.max_episode_length:
            self.done = True
        a = self.a_max * action    # The input 'action' lives in [-1,1]^d
        
        rew = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            u = self.x[i] ** 2 + np.sum(a ** 2) - a[i] ** 2
            v = np.sum(self.x ** 2) - self.x[i] ** 2 + a[i] ** 2
            rew[i] = - (1 - self.xi) * u - self.xi * v
            
        self.x += a + self.noise_std * np.random.randn(self.n_obj)
        
        return self.normalizer(deepcopy(self.x)), rew, \
               deepcopy(self.done), {}



