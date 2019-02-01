from abstract import DecayFunction
from noises import Noise, ActionNoise, ParameterNoise

def setup_noise(
        noise_type: str, sigma: float, theta: float, dt: float,
        sigma_decay: DecayFunction, noscale: bool, **kwargs) -> Noise:
    keywords_args = dict(sigma=sigma, theta=theta, dt=dt, noscale=noscale,
                         sigma_decay=sigma_decay)

    if noise_type == 'parameter':
        return ParameterNoise(**keywords_args) # type: ignore
    elif noise_type == 'action':
        return ActionNoise(**keywords_args) # type: ignore
    else:
        raise ValueError("Incorrect noise type...")


