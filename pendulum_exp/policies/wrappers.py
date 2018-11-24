"""Policy wrappers"""
from typing import Dict, Optional
from abstract import Policy, StateDict, Arrayable
from convert import check_array
import numpy as np

class StateNormalization(Policy):
    def __init__(self, policy: Policy, state: Dict[str, np.ndarray]) -> None:
        self._policy = policy
        self._count = 0
        self._state = state
        self._train = True

    def _normalize(self, obs: Arrayable) -> Arrayable:
        obs = check_array(obs)
        if self._train:
            new_count = self._count + obs.shape[0]
            if self._state['mean'] is None or self._state['mean_squares'] is None:
                self._state['mean'] = obs.sum(axis=0) / new_count
                self._state['mean_squares'] = (obs ** 2).sum(axis=0) / new_count
            else:
                self._state['mean'] = (self._state['mean'] * self._count + obs.sum(axis=0)) / new_count
                self._state['mean_squares'] = (self._state['mean_squares'] * self._count +
                                               (obs ** 2).sum(axis=0)) / new_count
            self._count = new_count
        std = np.sqrt(self._state['mean_squares'] - self._state['mean'] ** 2) + 1e-5
        normalized_obs = (obs - self._state['mean'][np.newaxis, ...]) / std[np.newaxis, ...]
        return np.clip(normalized_obs, -10, 10)

    def step(self, obs: Arrayable):
        obs = self._normalize(obs)
        return self._policy.step(obs)

    def observe(self,
                next_obs: Arrayable,
                reward: Arrayable,
                done: Arrayable,
                time_limit: Optional[Arrayable]):
        next_obs = self._normalize(next_obs)
        return self._policy.observe(
            next_obs, reward, done, time_limit)

    def learn(self):
        return self._policy.learn()

    def reset(self):
        return self._policy.reset()

    def train(self):
        self._train = True
        return self._policy.train()

    def eval(self):
        self._train = False
        return self._policy.eval()

    def load_state_dict(self, state_dict: StateDict):
        self._state['mean'] = state_dict['obs_mean']
        self._state['mean_squares'] = state_dict['obs_mean_squares']
        return self._policy.load_state_dict(state_dict)

    def state_dict(self) -> StateDict:
        state_dict = self._policy.state_dict()
        state_dict['obs_mean'] = self._state['mean']
        state_dict['obs_mean_squares'] = self._state['mean_squares']
        return state_dict

    def value(self, obs: Arrayable):
        assert hasattr(self._policy, 'value')
        obs = self._normalize(obs)
        return self._policy.value(obs) # type: ignore

    def advantage(self, obs: Arrayable, action: Arrayable):
        assert hasattr(self._policy, 'advantage')
        obs = self._normalize(obs)
        return self._policy.advantage(obs) # type: ignore

    def observe_evaluation(self, eval_return: float):
        self._policy.observe_evaluation(eval_return)
