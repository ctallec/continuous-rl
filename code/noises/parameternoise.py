from typing import Callable, Optional, Dict, Any
import numpy as np
import torch
from abstract import Arrayable, ParametricFunction
from noises.noise import Noise
from convert import th_to_arr



class ParameterNoise(Noise):
    """ Ornstein Ulhenbeck parameter noise. """
    def __init__(self,
                 theta: float,
                 sigma: float,
                 dt: float,
                 noscale: bool,
                 sigma_decay: Optional[Callable[[int], float]] = None) -> None:
        self._p_noise: Dict[str, Any] = {}
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
        for name in self._p_noise:
            self._p_noise[name] = self._p_noise[name].to(device)
        return self

    def step(self):
        """ Perform one step of update of parameter noise. """
        for _, p in self._p_noise.items():
            decay = 1 if self._sigma_decay is None else self._sigma_decay(self._count)
            dBt = torch.randn_like(p, requires_grad=False) * self._sigma * \
                decay * np.sqrt(self._dt)
            p.copy_(p * (1 - self._theta * self._dt) + dBt)
        self._count += 1

    def __iter__(self):
        return iter(self._p_noise)

    def _init_noise(self, function: ParametricFunction):
        self._p_noise = {
            name: self._sigma / np.sqrt(2 * self._theta) * torch.randn_like(p, requires_grad=False)
            for name, p in function.named_parameters() if 'ln' not in name}

    def perturb_output(
            self,
            *inputs: Arrayable,
            function: ParametricFunction):
        if not self._p_noise:
            self._init_noise(function)

        with torch.no_grad():
            for name, p in function.named_parameters():
                if 'ln' not in name:
                    p.copy_(p.data + self._p_noise[name])
            perturbed_output = function(*inputs)
            for name, p in function.named_parameters():
                if 'ln' not in name:
                    p.copy_(p.data - self._p_noise[name])
            return th_to_arr(perturbed_output)