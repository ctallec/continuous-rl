"""
Implement simple hill environment.
"""
import gym
from gym.utils import seeding
from gym.spaces import Discrete, Box
import numpy as np
from abstract import Env

class HillEnv(gym.Env, Env):
    def __init__(self):
        self.observation_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.action_space = Discrete(2)
        self._x = None
        self.dt = .1
        self.seed()

    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        if self._x[0] < 0:
            done = True
        elif self._x[0] < 10 * self.dt:
            self._x = self._x - self.dt
        else:
            self._x += 2 * (action - 1) * self.dt

        reward = 1

        return self._x, reward, done, {}

    def reset(self):
        np.random.seed()
        self._x = np.clip(np.random.normal(0, 1, (1,)), self.dt, 1)
        return self._x

    def close(self):
        pass

    def render(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
