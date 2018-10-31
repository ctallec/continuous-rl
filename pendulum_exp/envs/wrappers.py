"""Env wrappers"""
from gym import ActionWrapper
from gym.spaces import Discrete, Box
import numpy as np

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
