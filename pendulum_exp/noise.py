""" Noise. """
from typing import Callable, Optional, Dict, Any
import numpy as np
import torch
from abstract import Noise, Arrayable, ParametricFunction, DecayFunction, Tensorable
from convert import th_to_arr, check_tensor

def setup_noise(
        noise_type: str, sigma: float, theta: float, dt: float,
        sigma_decay: DecayFunction, **kwargs) -> Noise:
    keywords_args = dict(sigma=sigma, theta=theta, dt=dt, sigma_decay=sigma_decay)

    if noise_type == 'parameter':
        return ParameterNoise(**keywords_args) # type: ignore
    elif noise_type == 'action':
        return ActionNoise(**keywords_args) # type: ignore
    else:
        raise ValueError("Incorrect noise type...")

class ParameterNoise(Noise):
    """ Ornstein Ulhenbeck parameter noise. """
    def __init__(self,
                 theta: float,
                 sigma: float,
                 dt: float,
                 sigma_decay: Optional[Callable[[int], float]] = None) -> None:
        self._p_noise: Dict[str, Any] = {}
        self._theta = theta
        self._sigma = sigma
        self._sigma_decay = sigma_decay
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

class ActionNoise(Noise): # pylint: disable=too-few-public-methods
    """ Ornstein Ulhenbeck action noise. """
    def __init__(self, # pylint: disable=too-many-arguments
                 theta: float,
                 sigma: float,
                 dt: float,
                 sigma_decay: Optional[Callable[[int], float]] = None) -> None:
        self.noise = torch.empty(())
        self._theta = theta
        self._sigma = sigma
        self._sigma_decay = sigma_decay
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
