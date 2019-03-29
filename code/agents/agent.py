"""Define abstractions."""
from typing import Optional
from abc import abstractmethod

from numpy import ndarray as array
from torch import Tensor

from abstract import Arrayable
from stateful import Stateful

class Agent(Stateful):
    @abstractmethod
    def step(self, obs: Arrayable) -> array:
        """Returns an array of action given an Arrayable of observations.

        This method behaves differently depending on wether you are training or
        evaluating.
        """
        pass

    @abstractmethod
    def observe(self,
                next_obs: Arrayable,
                reward: Arrayable,
                done: Arrayable,
                time_limit: Optional[Arrayable]) -> None:
        """Observe a transition.

        This is where learn is called in most instances.
        Can include storing memory in a buffer.
        """
        pass

    @abstractmethod
    def value(self, obs: Arrayable) -> Tensor:
        """Returns estimated value of the given arrayable of observations."""
        pass

    @abstractmethod
    def actions(self, obs: Arrayable) -> Tensor:
        """Returns a tensor representative of the action taken when observing obs.

        This is not necessarily the action taken, but can be a proxy to this action
        (typically the probability of the action instead of the action itself.
        """
        pass

    @abstractmethod
    def learn(self) -> None:
        """Perform learning."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Perform reset operations."""
        pass

    @abstractmethod
    def train(self) -> None:
        """Change mode to train mode."""
        pass

    @abstractmethod
    def eval(self) -> None:
        """Change mode to eval mode."""
        pass
