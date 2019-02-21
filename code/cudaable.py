"""Abstraction for things attachable to a cuda capable device."""
from abc import ABC, abstractmethod

class Cudaable(ABC):
    @abstractmethod
    def to(self, device):
        raise NotImplementedError()
