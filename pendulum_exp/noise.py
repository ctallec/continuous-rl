""" Noise. """
from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn # pylint: disable=useless-import-alias

class ParameterNoise: # pylint: disable=too-few-public-methods
    """ Ornstein Ulhenbeck parameter noise. """
    def __init__(self, # pylint: disable=too-many-arguments
                 network: nn.Module,
                 theta: float,
                 sigma: float,
                 dt: float,
                 sigma_decay: Optional[Callable[[int], float]] = None) -> None:
        self._p_noise = [
            sigma / np.sqrt(2 * theta) * torch.randn_like(p, requires_grad=False)
            for p in network.parameters()]
        self._theta = theta
        self._sigma = sigma
        self._sigma_decay = sigma_decay
        self._dt = dt
        self._count = 0


    def step(self):
        """ Perform one step of update of parameter noise. """
        for p in self._p_noise:
            decay = 1 if self._sigma_decay is None else self._sigma_decay(self._count)
            dBt = torch.randn_like(p, requires_grad=False) * self._sigma * \
                decay * np.sqrt(self._dt)
            p.copy_(p * (1 - self._theta * self._dt) + dBt)
        self._count += 1

    def __iter__(self):
        return iter(self._p_noise)
