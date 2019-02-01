"""Define abstractions."""
from abc import ABC, abstractmethod
from abstract import Arrayable

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