""" Noise. """
from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from abstract import Noise, Arrayable, ParametricFunction
from config import NoiseConfig, ParameterNoiseConfig, ActionNoiseConfig
from convert import th_to_arr

def setup_noise(
        noise_config: NoiseConfig,
        **kwargs) -> Noise:
    assert isinstance(noise_config, (ParameterNoise, ActionNoiseConfig))
    keywords_args = dict(sigma=noise_config.sigma, theta=noise_config.theta,
                         dt=noise_config.dt, sigma_decay=noise_config.sigma_decay)

    if isinstance(noise_config, ParameterNoiseConfig):
        assert kwargs['network'] is not None
        keywords_args['network'] = kwargs['network']
        return ParameterNoise(**keywords_args) # type: ignore
    if isinstance(noise_config, ActionNoiseConfig):
        assert kwargs['action_shape'] is not None
        keywords_args['action_shape'] = kwargs['action_shape']
        return ActionNoise(**keywords_args) # type: ignore

class ParameterNoise(Noise):
    """ Ornstein Ulhenbeck parameter noise. """
    def __init__(self,
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
        self._device = torch.device('cpu')

    def to(self, device):
        self._device = device

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

    def perturb_output(
            self,
            inputs: Arrayable,
            a_function: ParametricFunction):
        with torch.no_grad():
            for p, p_noise in zip(a_function.parameters(), self):
                p.copy_(p.data + p_noise)
            perturbed_output = a_function(inputs)
            for p, p_noise in zip(a_function.parameters(), self):
                p.copy_(p.data - p_noise)
            return th_to_arr(perturbed_output)

class ActionNoise(Noise): # pylint: disable=too-few-public-methods
    """ Ornstein Ulhenbeck action noise. """
    def __init__(self, # pylint: disable=too-many-arguments
                 action_shape: Tuple[int, int],
                 theta: float,
                 sigma: float,
                 dt: float,
                 sigma_decay: Optional[Callable[[int], float]] = None) -> None:
        self.noise = sigma / np.sqrt(2 * theta) * \
            torch.randn(action_shape, requires_grad=False)
        self._theta = theta
        self._sigma = sigma
        self._sigma_decay = sigma_decay
        self._dt = dt
        self._count = 0
        self._device = torch.device('cpu')

    def to(self, device):
        self._device = device

    def step(self):
        """ Perform one step of update of parameter noise. """
        decay = 1 if self._sigma_decay is None else self._sigma_decay(self._count)
        dBt = torch.randn_like(self.noise, requires_grad=False) * self._sigma * \
            decay * np.sqrt(self._dt)
        self.noise = self.noise * (1 - self._theta * self._dt) + dBt
        self._count += 1

    def perturb_output(
            self,
            inputs: Arrayable,
            function: ParametricFunction):
        with torch.no_grad():
            return function(inputs) + self.noise
