"""Define abstractions."""
from typing import Union, Callable, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from torch import Tensor
import numpy as np

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

class Loggable(ABC):
    @abstractmethod
    def log(self):
        raise NotImplementedError()
