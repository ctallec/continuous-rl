"""Define abstractions."""
from abc import abstractmethod
from torch import Tensor

from abstract import Loggable, Arrayable
from cudaable import Cudaable
from stateful import Stateful

class Actor(Stateful, Cudaable, Loggable):
    """Abstract class for actors."""
    @abstractmethod
    def act(self, obs: Arrayable, target: bool = False) -> Tensor:
        """Perform action after observing observations obs.

        :args obs: observations to react to
        :args target: if True, may use target network

        :return: tensor of actions
        """
        raise NotImplementedError()

    @abstractmethod
    def act_noisy(self, obs: Arrayable) -> Arrayable:
        """Perform noisy action after observing obs."""
        raise NotImplementedError()

    @abstractmethod
    def optimize(self, loss: Tensor) -> None:
        """Perform one step of optimization.

        :args loss: either 1D (batch,) or 0D tensor
        """
        raise NotImplementedError()
