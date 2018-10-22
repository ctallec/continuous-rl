"""Define some configuration facitilies."""
from abc import ABCMeta
from typing import NamedTuple, Callable

class NoiseConfig:
    __metaclass__ = ABCMeta

class ParameterNoiseConfig(NamedTuple):
    sigma: float
    theta: float
    dt: float
    sigma_decay: Callable[[int], float]

class ActionNoiseConfig(NamedTuple):
    sigma: float
    theta: float
    dt: float
    sigma_decay: Callable[[int], float]


NoiseConfig.register(ParameterNoiseConfig) # type: ignore
NoiseConfig.register(ActionNoiseConfig) # type: ignore
