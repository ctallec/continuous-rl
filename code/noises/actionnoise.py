from typing import Callable, Optional
import numpy as np
import torch
from abstract import Arrayable, ParametricFunction, Tensorable
from noises.noise import Noise
from convert import th_to_arr, check_tensor




class ActionNoise(Noise): # pylint: disable=too-few-public-methods
    """ Ornstein Ulhenbeck action noise. """
    def __init__(self, # pylint: disable=too-many-arguments
                 theta: float,
                 sigma: float,
                 dt: float,
                 noscale: bool,
                 sigma_decay: Optional[Callable[[int], float]] = None) -> None:
        self.noise = torch.empty(())
        self._theta = theta
        self._sigma = sigma
        self._sigma_decay = sigma_decay
        ref_dt = .02
        if noscale:
            self._dt = ref_dt
        else:
            self._dt = dt
        self._count = 0
        self._device = torch.device('cpu')

    def to(self, device):
        self._device = device
        self.noise = self.noise.to(device)
        return self

    def step(self):
        """ Perform one step of update of parameter noise. """
        decay = 1 if self._sigma_decay is None else self._sigma_decay(self._count)
        dBt = torch.randn_like(self.noise, requires_grad=False) * self._sigma * \
            decay * np.sqrt(self._dt)
        self.noise = self.noise * (1 - self._theta * self._dt) + dBt
        self._count += 1

    def _init_noise(self, template: Tensorable):
        action_shape = check_tensor(template).size()
        self.noise = self._sigma / np.sqrt(2 * self._theta) * \
            torch.randn(action_shape, requires_grad=False).to(self._device)

    def perturb_output(
            self,
            *inputs: Arrayable,
            function: ParametricFunction):
        with torch.no_grad():
            output = function(*inputs)
            if not self.noise.shape:
                self._init_noise(output)
            return th_to_arr(output[:self.noise.size(0)] + self.noise)
