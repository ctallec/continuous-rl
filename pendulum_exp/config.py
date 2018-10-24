"""Define some configuration facitilies."""
from typing import NamedTuple, Callable, Union


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


NoiseConfig = Union[ParameterNoiseConfig, ActionNoiseConfig]
