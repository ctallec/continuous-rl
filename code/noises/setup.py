from abstract import DecayFunction
from noises import Noise, ActionNoise

def setup_noise(
        noise_type: str, sigma: float, theta: float, dt: float,
        sigma_decay: DecayFunction, noscale: bool, **kwargs) -> Noise:
    keywords_args = dict(sigma=sigma, theta=theta, dt=dt, noscale=noscale,
                         sigma_decay=sigma_decay)

    if noise_type == 'coherent':
        return ActionNoise(**keywords_args) # type: ignore
    elif noise_type == 'independant':
        raise ValueError("Incorrect noise type...")
    else:
        raise ValueError("Incorrect noise type...")
