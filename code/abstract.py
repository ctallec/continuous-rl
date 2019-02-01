"""Define abstractions."""
from typing import Union, Callable, Tuple
from abc import ABC, abstractmethod
from torch import Tensor
import numpy as np

from stateful import Stateful
from cudaable import Cudaable

Arrayable = Union[list, float, np.ndarray]
Tensorable = Union[Arrayable, Tensor]
DecayFunction = Callable[[int], float]
Shape = Tuple[Tuple[int, ...], ...]


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
