"""Define abstractions."""
from abc import abstractmethod
from torch import Tensor


from abstract import Stateful, Cudaable, Loggable, Arrayable


class Actor(Stateful, Cudaable, Loggable):
    @abstractmethod
    def act(self, obs: Arrayable, target: bool = False) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def act_noisy(self, obs: Arrayable) -> Arrayable:
        raise NotImplementedError()

    @abstractmethod
    def optimize(self, loss: Tensor):
        raise NotImplementedError()