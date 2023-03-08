import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Optional, Callable, Tuple, List
import gym
from numba import jit, float64, int64
from numba.experimental import jitclass




@jit(float64(float64))
def sigmoid(x: float):
    return 1 / (1 + np.exp(-x)) if x >= 0 else 1 - 1 / (np.exp(x) + 1)


@jit(float64(float64, int64))
def recovery_function(rt: float, t: int):
    assert rt > 0 and t >= 0
    return 1 - np.exp(- t / rt)    # must be 0 at t = 0


@jitclass([('w', float64), ('b', float64), ('c', float64), ('rt', float64)])
class Response:
    def __init__(self, w: float, b: float, c: float, recovery_time: float):
        assert c > 0 and recovery_time > 0
        self.w = w
        self.b = b
        self.c = c
        self.rt = recovery_time
    
    def call(self, price: float, t: int):
        return self.c * sigmoid(self.w * price + self.b) * recovery_function(self.rt, t)


class Group:
    def __init__(self, size: int, purchase_prob: Response):
        assert size >= 1
        self.purchase_prob = purchase_prob
        self.size = size    # number of customers in this group
        self.t_since_last_purchase = [9999] * self.size
        
    def offered(self, price: float):
        assert 0 <= price <= 1
        decisions = np.zeros(self.size, dtype=int)
        for i in range(self.size):
            if np.random.uniform() < self.purchase_prob.call(price, self.t_since_last_purchase[i]):
                decisions[i] = 1
                self.t_since_last_purchase[i] = 0    # buy
            else:
                self.t_since_last_purchase[i] += 1    # reject
        return decisions


class Market(gym.Env):
    def __init__(self, memory_length: int, group_size: int, 
                 normalizer: Optional[Callable] = None):
        super().__init__()
        if normalizer is None:
            self.normalizer = lambda x: x
        else:
            self.normalizer = lambda x: normalizer(x)
            
        rs = [Response(w=-8, b=7, c=0.9, recovery_time=15.0), 
              Response(w=0, b=12, c=0.5, recovery_time=9.0), 
              Response(w=-9, b=4.5, c=1, recovery_time=6.0), 
              Response(w=-10, b=3, c=0.9, recovery_time=12.0)]

        group_sizes = [group_size] * 4
        self.groups = [Group(size=group_sizes[i], purchase_prob=rs[i]) for i in range(len(rs))]
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                           shape=(len(self.groups),))   # prices offered to each group
        self.memory_length = memory_length
        
    def set_ep_len(self, max_episode_length: int):
        self.max_episode_length = max_episode_length
        return self
        
    def reset(self):
        self.done = False
        self.t = 0
        self.state = {'offered_prices': 1.1 * np.ones((self.memory_length, len(self.groups))), 
                      'acceptance_rate': np.zeros((self.memory_length, len(self.groups)))}
        return self.normalizer(self.state)
    
    @staticmethod
    def _fairness(action: NDArray) -> float:
        return - np.std(action)
    
    def _reward(self, action: NDArray, outcome: ArrayLike) -> NDArray:
        assert len(action) == len(self.groups)
        revenue = [np.sum(outcome[i]) * action[i] for i in range(len(self.groups))]
        n_all_customers = np.sum([g.size for g in self.groups])
        fairness_score = self._fairness(action)
        return np.array([np.sum(revenue) / n_all_customers, fairness_score])
    
    def step(self, action: NDArray) -> Tuple:
        assert self.max_episode_length is not None
        assert not self.done
        assert np.shape(action) == (len(self.groups),)
        assert np.max(action) <= 1 and np.min(action) >= 0
        self.t += 1
        if self.t >= self.max_episode_length:
            self.done = True
        
        outcome = [g.offered(action[i]) for i, g in enumerate(self.groups)]
        rew = self._reward(action, outcome)
        a_rate = [np.mean(outcome[i]) for i in range(len(outcome))]
        
        next_state = {'offered_prices': np.vstack([self.state['offered_prices'], action])[1:],
                      'acceptance_rate': np.vstack([self.state['acceptance_rate'], a_rate])[1:]}
        self.state = next_state

        return self.normalizer(self.state), rew, self.done, {}


