"""
Implement simple hill environment.
"""
import gym
from gym.utils import seeding
from gym.spaces import Discrete, Box
import numpy as np
from envs.env import Env

class HillEnv(gym.Env, Env):
    def __init__(self, discrete: bool=True) -> None:
        if discrete:
            self._action_space = Discrete(2)
        else:
            self._action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self._observation_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self._x = None
        self.dt = .1
        self.seed()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def action(self, action):
        if isinstance(self._action_space, Box):
            return action
        else:
            return 2 * action - 1

    def step(self, action):
        done = False

        # perform action
        if -1 < self._x[0] < -.25:
            # if -1 < x < -.25, action has no impact
            # we go left
            self._x[0] = self._x[0] - self.dt
        elif self._x[0] >= 1.:
            self._x[0] = 1.
        else:
            self._x += self.action(action) * self.dt

        # compute reward: if x < 0, r = 1, 0 otherwise
        if self._x[0] < 0:
            reward = 1
        else:
            reward = 0

        # compute termination: if x <= -1, done
        if self._x[0] <= -1:
            done = True

        return self._x, reward, done, {}

    def reset(self):
        np.random.seed()
        self._x = np.clip(np.random.normal(0, 1, (1,)), 0, 1)
        return self._x

    def close(self):
        pass

    def render(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
