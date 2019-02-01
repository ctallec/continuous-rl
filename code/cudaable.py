from abc import ABC, abstractmethod

class Cudaable(ABC):
    @abstractmethod
    def to(self, device):
        raise NotImplementedError()

