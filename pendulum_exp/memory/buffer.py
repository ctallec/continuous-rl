"""Define memory sampler"""
from typing import Optional, Tuple
import numpy as np
from abstract import Arrayable
from convert import check_array

from memory.sumtree import SumTree

class MemorySampler:
    def __init__(self, size: int, batch_size: int) -> None:
        self._size = size
        self._true_size = -1
        self._batch_size = batch_size
        self._full = False
        self._cur = 0

        # delay buffer initialization
        self._obs = np.empty(0)
        self._action = np.empty(0)
        self._next_obs = np.empty(0)
        self._reward = np.empty(0)
        self._done = np.empty(0)

        # store a reference point for relative td
        self._ref_obs = np.empty(0)

    @property
    def size(self):
        return self._true_size

    def push(
            self,
            obs: Arrayable,
            action: Arrayable,
            next_obs: Arrayable,
            reward: Arrayable,
            done: Arrayable) -> None:
        # if empty, initialize  buffer
        obs = check_array(obs)
        action = check_array(action)
        next_obs = check_array(next_obs)
        reward = check_array(reward)
        done = check_array(done)

        nb_envs = obs.shape[0]
        if self._true_size == -1:
            self._true_size = nb_envs * self._size
            self._obs = np.zeros((self._true_size, *obs.shape[1:]))
            self._action = np.zeros((self._true_size, *action.shape[1:]))
            self._next_obs = np.zeros((self._true_size, *next_obs.shape[1:]))
            self._reward = np.zeros((self._true_size, *reward.shape[1:]))
            self._done = np.zeros((self._true_size, *done.shape[1:]))

            # initialize reference point
            self._ref_obs = obs.copy()

        self._obs[self._cur:self._cur + nb_envs] = obs
        self._action[self._cur:self._cur + nb_envs] = action
        self._next_obs[self._cur:self._cur + nb_envs] = next_obs
        self._reward[self._cur:self._cur + nb_envs] = reward
        self._done[self._cur:self._cur + nb_envs] = done
        if self._cur + nb_envs == self._true_size:
            self._full = True
        self._cur = (self._cur + nb_envs) % (self._true_size)

    def sample(self, idxs: Optional[Arrayable]=None, to_observe: bool=True) -> Tuple[Arrayable, ...]:
        size = self._true_size if self._full else self._cur
        if idxs is None:
            idxs = np.random.randint(0, size, self._batch_size)
        return (
            self._obs[idxs], self._action[idxs], self._next_obs[idxs],
            self._reward[idxs], self._done[idxs], 1.)

    def observe(self, priorities: Arrayable):
        pass

    @property
    def reference_obs(self):
        if self._true_size == -1:
            raise IndexError()
        return self._ref_obs

class PrioritizedMemorySampler:
    def __init__(self, size: int, batch_size: int,
                 beta: float, alpha: float) -> None:
        self._memory = MemorySampler(size, batch_size)
        self._sum_tree = SumTree(size)
        self._max_priority = .1
        self._batch_size = batch_size
        self._beta = beta
        self._alpha = alpha

        # placeholders to update priorities
        self._idxs = None

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        assert 0. <= beta <= 1.
        self._beta = beta

    def push(
            self,
            obs: Arrayable,
            action: Arrayable,
            next_obs: Arrayable,
            reward: Arrayable,
            done: Arrayable) -> None:
        self._memory.push(
            obs, action, next_obs, reward, done)
        for _ in check_array(obs):
            self._sum_tree.add(self._max_priority ** self._alpha)

    def sample(self, to_observe: bool=True) -> Tuple[Arrayable, ...]:
        if to_observe:
            assert self._idxs is None, "No observe after sample ..."
        idxs, priorities = zip(*[self._sum_tree.sample() for _ in range(self._batch_size)])
        idxs, priorities = check_array(idxs), check_array(priorities)
        obs, action, next_obs, reward, done, _ = self._memory.sample(idxs)
        weights = (self._sum_tree.total / self._memory.size / priorities) ** self._beta
        weights = weights / weights.max()

        if to_observe:
            self._idxs = idxs

        return obs, action, next_obs, reward, done, weights

    def observe(self, priorities: Arrayable):
        assert self._idxs is not None, "No sample before observe ..."
        self._max_priority = max(self._max_priority, priorities.max())
        for idx, prio in zip(self._idxs, priorities):
            self._sum_tree.modify(idx, prio ** self._alpha)

        self._idxs = None

    @property
    def reference_obs(self):
        return self._memory.reference_obs
