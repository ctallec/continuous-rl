from gym import Space
from gym.spaces import Discrete, Box

from abstract import DecayFunction
from noises import Noise, CoherentNoise
from noises import IndependentContinuousNoise, IndependentDiscreteNoise

def setup_noise(
        noise_type: str, sigma: float, theta: float, epsilon: float, dt: float,
        sigma_decay: DecayFunction, noscale: bool, action_space: Space,
        **kwargs) -> Noise:
    if noise_type == 'coherent':
        return CoherentNoise(theta, sigma, dt, noscale, sigma_decay)
    elif noise_type == 'independent':
        if isinstance(action_space, Discrete):
            return IndependentDiscreteNoise(epsilon, sigma_decay)
        elif isinstance(action_space, Box):
            return IndependentContinuousNoise(sigma, sigma_decay)
        raise ValueError("Incorrect noise type...")
    else:
        raise ValueError("Incorrect noise type...")
