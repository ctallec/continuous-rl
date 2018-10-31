"""Env wrappers"""
import gym
from gym import ActionWrapper
from gym.spaces import Discrete, Box
import numpy as np
from convert import check_array
from abstract import Arrayable

class WrapPendulum(ActionWrapper):
    """ Wrap pendulum. """
    @property
    def action_space(self):
        return Discrete(2)

    @action_space.setter
    def action_space(self, value):
        self.env.action_space = value

    def action(self, action):
        return 4 * np.array(action)[np.newaxis] - 2

class WrapContinuousPendulum(ActionWrapper):
    """ Wrap Continuous Pendulum. """
    @property
    def action_space(self):
        return Box(low=-1, high=1, shape=(1,))

    @action_space.setter
    def action_space(self, value):
        self.env.action_space = value

    def action(self, action):
        return np.clip(2 * action, -2, 2)

class ObservationNormalizedWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.env.observation_space, Box)
        self.observation_space = Box(low=-10, high=10,
                                     shape=self.env.observation_space.shape,
                                     dtype=np.float32)
        self._count = 0
        self._mean = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        self._mean_squared = np.zeros(self.env.observation_space.shape, dtype=np.float32)

    def observation(self, observation: Arrayable):
        observation = check_array(observation)
        new_count = self._count + observation.shape[0]
        self._mean = (self._mean * self._count + observation.sum(axis=0)) / new_count
        self._mean_squared = (self._mean_squared * self._count +
                              (observation ** 2).sum(axis=0)) / new_count
        self._count = new_count
        std = np.sqrt(self._mean_squared - self._mean ** 2)
        normalized_obs = (observation - self._mean[np.newaxis, ...]) / std[np.newaxis, ...]
        return np.clip(normalized_obs, -10, 10)
