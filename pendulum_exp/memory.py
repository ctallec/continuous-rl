"""Define memory sampler"""
import numpy as np
from abstract import Arrayable
from convert import check_array

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

        self._obs[self._cur:self._cur + nb_envs] = obs
        self._action[self._cur:self._cur + nb_envs] = action
        self._next_obs[self._cur:self._cur + nb_envs] = next_obs
        self._reward[self._cur:self._cur + nb_envs] = reward
        self._done[self._cur:self._cur + nb_envs] = done
        if self._cur + nb_envs == self._true_size:
            self._full = True
        self._cur = (self._cur + nb_envs) % (self._true_size)

    def sample(self) -> np.ndarray:
        size = self._true_size if self._full else self._cur
        idxs = np.random.randint(0, size, self._batch_size)
        return (
            self._obs[idxs], self._action[idxs], self._next_obs[idxs],
            self._reward[idxs], self._done[idxs])
