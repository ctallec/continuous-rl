"""Generalities for parametric policies."""
from typing import Any
from abstract import Policy, ParametricFunction, StateDict, Arrayable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class ParametricPolicy(Policy):
    """Torch like registering of net, optimizer and scheduler."""
    def __init__(self):
        self._optim_state = {}
        self._net_state = {}
        self._schedule_state = {}

    def __setattr__(self, key: str, value: Any):
        if isinstance(value, Optimizer):
            self._optim_state[key] = value
        elif isinstance(value, ParametricFunction):
            self._net_state[key] = value
        elif isinstance(value, _LRScheduler):
            self._schedule_state[key] = value
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key: str) -> Any:
        if key in self._optim_state:
            return self._optim_state[key]
        elif key in self._net_state:
            return self._net_state[key]
        elif key in self._schedule_state:
            return self._schedule_state[key]

    def state_dict(self) -> StateDict:
        state: StateDict = {}
        state.update(self._optim_state)
        state.update(self._net_state)
        state.update(self._schedule_state)
        return state

    def load_state_dict(self, state_dict: StateDict):
        for k_opt in self._optim_state:
            self._optim_state[k_opt] = state_dict[k_opt]
        for k_net in self._net_state:
            self._net_state[k_net] = state_dict[k_net]
        for k_sched in self._schedule_state:
            self._schedule_state[k_sched] = state_dict[k_sched]

    def eval(self):
        raise NotImplementedError()

    def learn(self):
        raise NotImplementedError()

    def log_stats(self):
        raise NotImplementedError()

    def networks(self):
        raise NotImplementedError()

    def observe_evaluation(self, eval_return: float):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def step(self, obs: Arrayable):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def observe(self):
        raise NotImplementedError()
