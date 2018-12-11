"""Generalities for parametric policies."""
from typing import Any
from abstract import StateDict, Stateful

class CompoundStateful:
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


Stateful.register(CompoundStateful)
