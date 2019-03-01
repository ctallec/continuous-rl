from typing import Callable, Optional

from numpy import ndarray as array
import torch

from abstract import Arrayable, ParametricFunction
from noises.noise import Noise
from convert import th_to_arr

class IndependentContinuousNoise(Noise):
    """Independent action noise."""
    def __init__(self,
                 sigma: float,
                 sigma_decay: Optional[Callable[[int], float]] = None) -> None:
        self._sigma = sigma

    def to(self, device):
        return self

    def step(self):
        """ Do nothing."""
        pass

    def perturb_output(
            self,
            *inputs: Arrayable,
            function: ParametricFunction) -> array:
        with torch.no_grad():
            output = function(*inputs)
            output = output + self._sigma * torch.randn_like(output)
            return th_to_arr(output)

class IndependentDiscreteNoise(Noise):
    """Independent discrete noise (epsilon greedy)."""
    def __init__(self,
                 epsilon: float,
                 epsilon_decay: Optional[Callable[[int], float]] = None) -> None:
        self._epsilon = epsilon

    def to(self, device):
        return self

    def step(self):
        """Do nothing."""
        pass

    def perturb_output(
            self,
            *inputs: Arrayable,
            function: ParametricFunction) -> array:
        with torch.no_grad():
            output = function(*inputs)
            bs = output.size(0)
            mask = (
                torch.zeros(
                    [bs] + (output.dim() - 1) * [1],
                    device=output.device).uniform_() > self._epsilon).float()
            output = mask * output + torch.randn_like(output) * (1 - mask)
            return th_to_arr(output)
