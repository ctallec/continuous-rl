from torch import Tensor
from typing import Tuple

from abstract import ParametricFunction, Tensorable
from optimizer import setup_optimizer

from stateful import CompoundStateful
from memory.memorytrajectory import BatchTraj
from utils import values
from abc import abstractmethod


class OnlineCritic(CompoundStateful):
    def __init__(self, gamma: float, dt: float, lr: float, optimizer: str,
                 v_function: ParametricFunction) -> None:
        CompoundStateful.__init__(self)
        self._reference_obs: Tensor = None
        self._v_function = v_function
        # self._optimizer = setup_optimizer(self._v_function.parameters(),
        #                                   opt_name=optimizer, lr=lr, dt=dt,
        #                                   inverse_gradient_magnitude=dt,
        #                                   weight_decay=0)
        self._gamma = gamma ** dt
        self._device = 'cpu'
        self._dt = dt

    # @abstractmethod
    # def optimize(self, v: Tensor, v_target: Tensor) -> Tensor:
    #     pass

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
