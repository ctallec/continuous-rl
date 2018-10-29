""" Define shared elements between continuous and discrete. """
import numpy as np
from abstract import Policy, Arrayable
from config import PolicyConfig
from memory import MemorySampler

class SharedAdvantagePolicy(Policy):
    def __init__(self, policy_config: PolicyConfig) -> None:
        self._train = True
        self.reset()

        # parameters
        self._gamma = policy_config.gamma
        self._dt = policy_config.dt
        self._alpha = policy_config.alpha
        self._learn_per_step = policy_config.learn_per_step
        self._sampler = MemorySampler(policy_config.memory_size)

    def reset(self):
        # internals
        self._obs = np.array([])
        self._action = np.array([])
        self._reward = np.array([])
        self._next_obs = np.array([])
        self._done = np.array([])

    def act(self, obs: Arrayable):
        raise NotImplementedError

    def step(self, obs: Arrayable):
        if self._train:
            self._obs = obs

        action = self.act(obs)
        if self._train:
            self._action = action

        return action

    def observe(self,
                next_obs: Arrayable,
                reward: Arrayable,
                done: Arrayable):
        if self._train:
            self._next_obs = next_obs
            self._reward = reward
            self._done = done
            self._sampler.push(
                self._obs, self._action, self._next_obs, self._reward, self._done)
            self.learn()
