from logging import info
from typing import Tuple

from torch import Tensor

from abstract import ParametricFunction, Tensorable
from memory.trajectory import BatchTraj
from stateful import CompoundStateful
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
        info(f"setup> using OnlineCritic, the provided gamma and rewards are scaled,"
             f" actual values: gamma={gamma ** dt}, rewards=original_rewards * {dt}")

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
