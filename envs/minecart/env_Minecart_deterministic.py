# Code taken and modified from https://github.com/axelabels/DynMORL

import os
import json
import numpy as np
import gym
from typing import Callable


EPS_SPEED = 0.001  # Minimum speed to be considered in motion
HOME_X = .0
HOME_Y = .0
HOME_POS = (HOME_X, HOME_Y)

ROTATION = 10
MAX_SPEED = 1.

FUEL_MINE = -.05
FUEL_ACC = -.025
FUEL_IDLE = -0.005

CAPACITY = 1

ACT_MINE = 0
ACT_LEFT = 1
ACT_RIGHT = 2
ACT_ACCEL = 3
ACT_BRAKE = 4
ACT_NONE = 5
ACTIONS = ["Mine", "Left", "Right", "Accelerate", "Brake", "None"]
ACTION_COUNT = len(ACTIONS)

MINE_RADIUS = 0.14
BASE_RADIUS = 0.15

MINE_SCALE = 1.
BASE_SCALE = 1.
CART_SCALE = 1.

MARGIN = 0.16 * CART_SCALE

ACCELERATION = 0.0075 * CART_SCALE
DECELERATION = 1


class Mine(object):
    def __init__(self, ore_cnt, x, y, ore_reward):
        self.pos = np.array([x, y])
        self.ore_reward = ore_reward

    def distance(self, cart):
        return np.linalg.norm(cart.pos - self.pos)

    def mineable(self, cart):
        return self.distance(cart) <= MINE_RADIUS * MINE_SCALE * CART_SCALE

    def mine(self):
        return self.ore_reward


class Cart(object):
    def __init__(self, ore_cnt):
        self.ore_cnt = ore_cnt
        self.pos = np.array([HOME_X, HOME_Y])
        self.speed = 0
        self.angle = 45
        self.content = np.zeros(self.ore_cnt)
        self.departed = False  # Keep track of whether the agent has left the base

    def accelerate(self, acceleration):
        self.speed = np.clip(self.speed + acceleration, 0, MAX_SPEED)

    def rotate(self, rotation):
        self.angle = (self.angle + rotation) % 360

    def step(self):
        """
            Update cart's position, taking the current speed into account
            Colliding with a border at anything but a straight angle will cause
            cart to "slide" along the wall.
        """
        pre = np.copy(self.pos)
        if self.speed < EPS_SPEED:
            return False
        x_velocity = self.speed * np.cos(self.angle * np.pi / 180)
        y_velocity = self.speed * np.sin(self.angle * np.pi / 180)
        x, y = self.pos
        if y != 0 and y != 1 and (y_velocity > 0 + EPS_SPEED or
                                  y_velocity < 0 - EPS_SPEED):
            if x == 1 and x_velocity > 0:
                self.angle += np.copysign(ROTATION, y_velocity)
            if x == 0 and x_velocity < 0:
                self.angle -= np.copysign(ROTATION, y_velocity)
        if x != 0 and x != 1 and (x_velocity > 0 + EPS_SPEED or
                                  x_velocity < 0 - EPS_SPEED):
            if y == 1 and y_velocity > 0:
                self.angle -= np.copysign(ROTATION, x_velocity)

            if y == 0 and y_velocity < 0:
                self.angle += np.copysign(ROTATION, x_velocity)

        self.pos[0] = np.clip(x + x_velocity, 0, 1)
        self.pos[1] = np.clip(y + y_velocity, 0, 1)
        self.speed = np.linalg.norm(pre - self.pos)
        self.angle = self.angle % 360

        return True


class Minecart(gym.Env):
    def __init__(self,
                 mine_cnt=3,
                 ore_cnt=2,
                 capacity=CAPACITY,
                 normalizer: Callable = None):

        super().__init__()
        self.capacity = capacity
        self.ore_cnt = ore_cnt
        self.mine_cnt = mine_cnt
        self.mines = []
        for i in range(self.mine_cnt):
            self.mines.append(Mine(self.ore_cnt, 0.0, 0.0, [None, None]))
        self.cart = Cart(self.ore_cnt)
        if normalizer is None:
            self.normalizer = lambda x: x
        else:
            self.normalizer = normalizer
        self.done = False

        low = np.append(np.array([0, 0, 0, 0]), np.zeros(ore_cnt))
        high = np.append(np.array([1, 1, MAX_SPEED, 360]), np.ones(ore_cnt) * capacity)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.max_episode_length = None

    def obj_cnt(self):
        return self.ore_cnt + 1

    @staticmethod
    def from_json(filename):
        with open(filename) as f:
            data = json.load(f)
            
        minecart = Minecart(
                       ore_cnt=data["ore_cnt"],
                       mine_cnt=data["mine_cnt"],
                       capacity=data["capacity"]
                   )
        if "mines" in data:
            for mine_data, mine in zip(data["mines"], minecart.mines):
                mine.pos = np.array([mine_data["x"], mine_data["y"]])
                if "distributions" in mine_data:
                    mine.ore_reward = [dist[0] for dist in mine_data["distributions"]]
        return minecart
    
    def set_normalizer(self, normalizer: Callable):
        self.normalizer = normalizer
        return self
    
    def set_ep_len(self, max_episode_length: int):
        self.max_episode_length = max_episode_length
        return self

    def reset(self):
        self.cart.content = np.zeros(self.ore_cnt)
        self.cart.pos = np.array(HOME_POS)
        self.cart.speed = 0
        self.cart.angle = 45
        self.cart.departed = False
        self.done = False
        self.t = 0
        return self.normalizer(self.get_state())

    def step(self, action, frame_skip=4):
        """Perform the given action `frame_skip` times
         ["Mine", "Left", "Right", "Accelerate", "Brake", "None"]
        Arguments:
            action {int} -- Action to perform, ACT_MINE (0), ACT_LEFT (1), ACT_RIGHT (2), ACT_ACCEL (3), ACT_BRAKE (4) or ACT_NONE (5)

        Keyword Arguments:
            frame_skip {int} -- Repeat the action this many times (default: {1})

        Returns:
            tuple -- (state, reward, terminal) tuple
        """
        assert self.max_episode_length is not None
        assert not self.done
        assert action in range(ACTION_COUNT)

        reward = np.zeros(self.ore_cnt + 1)

        self.t += 1
        if self.t >= self.max_episode_length:
            self.done = True

        reward[-1] = FUEL_IDLE * frame_skip

        if action == ACT_ACCEL:
            reward[-1] += FUEL_ACC * frame_skip
        elif action == ACT_MINE:
            reward[-1] += FUEL_MINE * frame_skip
            
        for _ in range(frame_skip):
            if action == ACT_LEFT:
                self.cart.rotate(-ROTATION)
            elif action == ACT_RIGHT:
                self.cart.rotate(ROTATION)
            elif action == ACT_ACCEL:
                self.cart.accelerate(ACCELERATION)
            elif action == ACT_BRAKE:
                self.cart.accelerate(-DECELERATION)
            elif action == ACT_MINE:
                self.mine()

            if self.done:
                break

            self.cart.step()

            distanceFromBase = np.linalg.norm(self.cart.pos - HOME_POS)
            if distanceFromBase < BASE_RADIUS * BASE_SCALE:
                if self.cart.departed:
                    # Cart left base then came back, ending the episode
                    self.done = True
                    # Sell resources
                    reward[: self.ore_cnt] += self.cart.content
                    self.cart.content = np.zeros(self.ore_cnt)
            else:
                # Cart left base
                self.cart.departed = True

        return self.normalizer(self.get_state()), reward.astype(np.float32), self.done, {}

    def mine(self):
        if self.cart.speed < EPS_SPEED:
            # Get closest mine
            mine = min(self.mines, key=lambda mine: mine.distance(self.cart))

            if mine.mineable(self.cart):
                cart_free = self.capacity - np.sum(self.cart.content)
                mined = mine.mine()
                total_mined = np.sum(mined)
                if total_mined > cart_free:
                    # Scale mined content to remaining capacity
                    scale = cart_free / total_mined
                    mined = np.array(mined) * scale
                self.cart.content += mined

    def get_state(self):
        return np.append(self.cart.pos,
                        [self.cart.speed, self.cart.angle, *self.cart.content]).astype(np.float32)

########################################################################################

def config_path(filename):
    return os.path.join(os.path.dirname(__file__), 'configs', filename)

def MinecartEnv():
    return Minecart.from_json(config_path('mine_config.json'))

def MinecartDeterministicEnv():
    return Minecart.from_json(config_path('mine_config_det.json'))

def MinecartSimpleDeterministicEnv():
    return Minecart.from_json(config_path('mine_config_1ore_det.json'))





