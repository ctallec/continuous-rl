"""Generalities for parametric policies."""
from typing import Any
from abc import ABC, abstractmethod
from typing import Dict
from cudaable import Cudaable

StateDict = Dict[str, Any]


class Stateful(ABC):
    """Abstraction for dict loadable/dumpable objects."""
    @abstractmethod
    def load_state_dict(self, state_dict: StateDict):
        raise NotImplementedError()

    @abstractmethod
    def state_dict(self) -> StateDict:
        raise NotImplementedError()


class CompoundStateful(Stateful, Cudaable):
    """Super class to automate registration of Statefuls.

    Each time an attribute subtype of Stateful is assigned to
    a subclass of CompoundStateful, the attribute is directly
    considered as part of the state, and will thus be dumped/loaded
    when calling load_state_dict and state_dict. Compound Stateful
    also make sure that Stateful attributes that are Cudaable are
    ported to the appropriate device when to is called.
    """
    def __init__(self):
        self._statefuls = {}

    def __setattr__(self, key: str, value: Any):
        if isinstance(value, Stateful):
            self._statefuls[key] = value
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__['_statefuls']:
            return self._statefuls[key]

    def unregister(self, key: str):
        assert key in self._statefuls
        self._statefuls[key] = None

    def state_dict(self) -> StateDict:
        return {k: v.state_dict() for (k, v) in self._statefuls.items()}

    def load_state_dict(self, state_dict: StateDict):
        for k in self._statefuls:
            self._statefuls[k].load_state_dict(state_dict[k])

    def to(self, device):
        for k in self._statefuls:
            if isinstance(self._statefuls[k], Cudaable):
                self._statefuls[k] = self._statefuls[k].to(device)
        return self
