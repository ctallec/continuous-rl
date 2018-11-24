"""Define abstractions."""
from typing import Union, Callable, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
import torch

Arrayable = Union[list, float, np.ndarray]
Tensorable = Union[Arrayable, torch.Tensor]
DecayFunction = Callable[[int], float]
StateDict = Dict[str, Any]

class Policy(ABC):
    @abstractmethod
    def step(self, obs: Arrayable):
        pass

    @abstractmethod
    def observe(self,
                next_obs: Arrayable,
                reward: Arrayable,
                done: Arrayable,
                time_limit: Optional[Arrayable]):
        pass

    @abstractmethod
    def observe_evaluation(self, eval_return: float):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: StateDict):
        pass

    @abstractmethod
    def state_dict(self) -> StateDict:
        pass

class ParametricFunction(ABC):
    """Wrap around a torch module."""
    @abstractmethod
    def __call__(self, *obs: Tensorable):
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def named_parameters(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: StateDict):
        pass

    @abstractmethod
    def state_dict(self) -> StateDict:
        pass

class Env(ABC):
    @abstractmethod
    def step(self, action: Arrayable):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self, mode='human'):
        pass

    @abstractmethod
    def close(self):
        pass

    @property
    @abstractmethod
    def observation_space(self):
        pass

    @property
    @abstractmethod
    def action_space(self):
        pass

class Noise(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def perturb_output(
            self,
            *inputs: Arrayable,
            function: ParametricFunction):
        pass

    @abstractmethod
    def to(
            self,
            device):
        pass
