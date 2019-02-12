from torch import Tensor
from typing import Tuple

from abstract import ParametricFunction, Tensorable

from stateful import CompoundStateful
from memory.memorytrajectory import BatchTraj
from utils import values

class OnlineCritic(CompoundStateful):
    def __init__(self, gamma: float, dt: float,
                 v_function: ParametricFunction) -> None:
        CompoundStateful.__init__(self)
        self._reference_obs: Tensor = None
        self._v_function = v_function
        self._gamma = gamma ** dt
        self._device = 'cpu'
        self._dt = dt

    def value(self, obs: Tensorable) -> Tensor:
        return self._v_function(obs)

    def value_batch(self, traj: BatchTraj) -> Tuple[Tensor, Tensor]:
        return values(self._v_function, traj, self._gamma, 1., self._dt)

    def log(self) -> None:
        pass

    def to(self, device) -> "OnlineCritic":
        CompoundStateful.to(self, device)
        self._device = device
        return self
