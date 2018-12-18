"""Define abstractions."""
from typing import Union, Callable, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from torch import Tensor

Arrayable = Union[list, float, np.ndarray]
Tensorable = Union[Arrayable, Tensor]
DecayFunction = Callable[[int], float]
StateDict = Dict[str, Any]
Shape = Tuple[Tuple[int, ...], ...]


class Stateful(ABC):
    @abstractmethod
    def load_state_dict(self, state_dict: StateDict):
        raise NotImplementedError()

    @abstractmethod
    def state_dict(self) -> StateDict:
        raise NotImplementedError()

class Cudaable(ABC):
    @abstractmethod
    def to(self, device):
        raise NotImplementedError()

class ParametricFunction(Stateful, Cudaable):
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
    def input_shape(self) -> Shape:
        pass

    @abstractmethod
    def output_shape(self) -> Shape:
        pass

class Policy(Stateful):
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

class Noise(Cudaable):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def perturb_output(
            self,
            *inputs: Arrayable,
            function: ParametricFunction):
        pass

class Loggable(ABC):
    @abstractmethod
    def log(self):
        raise NotImplementedError()

class Actor(Stateful, Cudaable, Loggable):
    @abstractmethod
    def act(self, obs: Arrayable, target: bool=False) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def act_noisy(self, obs: Arrayable) -> Arrayable:
        raise NotImplementedError()

    @abstractmethod
    def optimize(self, loss: Tensor):
        raise NotImplementedError()

class Critic(Stateful, Cudaable, Loggable):
    @abstractmethod
    def optimize(self, obs: Arrayable, action: Arrayable, max_action: Tensor,
                 next_obs: Arrayable, max_next_action: Tensor, reward: Arrayable,
                 done: Arrayable, time_limit: Arrayable, weights: Arrayable) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def critic(self, obs: Arrayable, action: Tensorable, target: bool=False) -> Tensor:
        raise NotImplementedError()
