"""Define some configuration facitilies."""
from abstract import DecayFunction
from typing import NamedTuple, Union, Tuple, Optional
import numpy as np


class ParameterNoiseConfig(NamedTuple):
    sigma: float
    theta: float
    dt: float
    sigma_decay: DecayFunction

class ActionNoiseConfig(NamedTuple):
    sigma: float
    theta: float
    dt: float
    sigma_decay: DecayFunction

class AdvantagePolicyConfig(NamedTuple):
    alpha: float
    gamma: float
    dt: float
    lr: float
    lr_decay: DecayFunction
    memory_size: int
    learn_per_step: int

class EnvConfig(NamedTuple):
    id: str
    dt: float


NoiseConfig = Union[ParameterNoiseConfig, ActionNoiseConfig]
PolicyConfig = AdvantagePolicyConfig

def read_config(
        args,
        config_file: Optional[str]=None) -> Tuple[
            PolicyConfig, NoiseConfig, NoiseConfig, EnvConfig]:
    # No configuration file at the moment
    def lr_decay(t):
        return 1 / np.power(1 + args.dt * t, 0)

    def noise_decay(_):
        return 1

    policy_config = AdvantagePolicyConfig(
        alpha=args.alpha, gamma=args.gamma, dt=args.dt,
        lr=args.lr, lr_decay=lr_decay, memory_size=args.memory_size, learn_per_step=args.learn_per_step)
    if args.noise_type == 'parameter':
        noise_config: NoiseConfig = ParameterNoiseConfig(
            args.sigma, args.theta, args.dt, noise_decay)
        eval_noise_config: NoiseConfig = ParameterNoiseConfig(
            args.sigma_eval, args.theta, args.dt, noise_decay)
    else:
        noise_config = ActionNoiseConfig(
            args.sigma, args.theta, args.dt, noise_decay)
        eval_noise_config = ActionNoiseConfig(
            args.sigma_eval, args.theta, args.dt, noise_decay)

    env_config = EnvConfig(args.env_id, args.dt)
    return policy_config, noise_config, eval_noise_config, env_config
