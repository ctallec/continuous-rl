"""Define abstractions."""
from typing import Optional
from abc import abstractmethod
from torch import Tensor
from stateful import Stateful

from abstract import Arrayable, Tensorable

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

    # Used for evaluation/logging
    @abstractmethod
    def value(self, obs: Arrayable) -> Tensor:
        pass

    @abstractmethod
    def actions(self, obs: Arrayable) -> Tensor:
        pass

    @abstractmethod
    def advantage(self, obs: Arrayable, action: Tensorable) -> Tensor:
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