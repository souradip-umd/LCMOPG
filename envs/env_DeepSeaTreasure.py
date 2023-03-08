import numpy as np
import gym
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
from numpy.typing import ArrayLike, NDArray



class DeepSeaTreasure(gym.Env):
    def __init__(self, convex: bool, noise_std: float, normalizer: Optional[Callable]):
        assert noise_std >= 0.0
        super().__init__()
        self.action_space = gym.spaces.Discrete(4,)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        
        if normalizer is None:
            self.normalizer = lambda x: x + noise_std * np.random.randn(len(x))
        else:
            self.normalizer = lambda x: normalizer(x) + noise_std * np.random.randn(len(x))
        self.max_episode_length = None
        
        if convex:
            self.sea_map = np.array(
                [[0,     0,    0,    0,    0,    0,    0,    0,    0,    0, 0],
                 [0.7,   0,    0,    0,    0,    0,    0,    0,    0,    0, 0],
                 [-10, 8.2,    0,    0,    0,    0,    0,    0,    0,    0, 0],
                 [-10, -10, 11.5,    0,    0,    0,    0,    0,    0,    0, 0],
                 [-10, -10,  -10, 14.0, 15.1, 16.1,    0,    0,    0,    0, 0],
                 [-10, -10,  -10,  -10,  -10,  -10,    0,    0,    0,    0, 0],
                 [-10, -10,  -10,  -10,  -10,  -10,    0,    0,    0,    0, 0],
                 [-10, -10,  -10,  -10,  -10,  -10, 19.6, 20.3,    0,    0, 0],
                 [-10, -10,  -10,  -10,  -10,  -10,  -10,  -10,    0,    0, 0],
                 [-10, -10,  -10,  -10,  -10,  -10,  -10,  -10, 22.4,    0, 0],
                 [-10, -10,  -10,  -10,  -10,  -10,  -10,  -10,  -10, 23.7, 0]], 
                 dtype=np.float64
            )
        else:
            # Original version by Vamplew et al., 2011
            self.sea_map = np.array(
                [[0,     0,   0,   0,   0,   0,   0,   0,   0,   0, 0],
                 [1,     0,   0,   0,   0,   0,   0,   0,   0,   0, 0],
                 [-10,   2,   0,   0,   0,   0,   0,   0,   0,   0, 0],
                 [-10, -10,   3,   0,   0,   0,   0,   0,   0,   0, 0],
                 [-10, -10, -10,   5,   8,  16,   0,   0,   0,   0, 0],
                 [-10, -10, -10, -10, -10, -10,   0,   0,   0,   0, 0],
                 [-10, -10, -10, -10, -10, -10,   0,   0,   0,   0, 0],
                 [-10, -10, -10, -10, -10, -10,  24,  50,   0,   0, 0],
                 [-10, -10, -10, -10, -10, -10, -10, -10,   0,   0, 0],
                 [-10, -10, -10, -10, -10, -10, -10, -10,  74,   0, 0],
                 [-10, -10, -10, -10, -10, -10, -10, -10, -10, 124, 0]],
                 dtype=np.float64
            )

        self.state_lim = [[0, 10], [0, 10]]

    def set_ep_len(self, max_episode_length: int):
        self.max_episode_length = max_episode_length
        return self
    
    def _get_map_value(self, pos: NDArray):
        return self.sea_map[pos[0]][pos[1]]

    def reset(self):
        self.state = np.array([0, 0])
        self.done = False
        self.t = 0
        return self.normalizer(deepcopy(self.state))

    def _is_inside_map(self, pos: NDArray):
        if (self.state_lim[0][0] <= pos[0] <= self.state_lim[0][1]) \
            and (self.state_lim[1][0] <= pos[1] <= self.state_lim[1][1]):
            return True
        else:
            return False
    
    def step(self, action: int):
        assert self.max_episode_length is not None
        assert not self.done
        assert isinstance(action, int), action
        self.t += 1
        
        move = {
            0: np.array([-1, 0]), 
            1: np.array([1, 0]), 
            2: np.array([0, -1]), 
            3: np.array([0, 1])
        }[action]
        next_state = self.state + move
        
        if self._is_inside_map(next_state) == False:
            rew = np.array([0.0, - 1.0])
            # no state transition
        else:
            rew = np.array([self._get_map_value(next_state), - 1.0])
            self.state = next_state
        
        if (rew[0] != 0.0) or (self.t >= self.max_episode_length):
            self.done = True
                
        return self.normalizer(deepcopy(self.state)), rew, deepcopy(self.done), {}




