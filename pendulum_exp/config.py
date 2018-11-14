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
    batch_size: int
    gamma: float
    dt: float
    lr: float
    lr_decay: DecayFunction
    memory_size: int
    learn_per_step: int
    steps_btw_train: int
    beta: Optional[float]

class SampledAdvantagePolicyConfig(NamedTuple):
    batch_size: int
    gamma: float
    dt: float
    lr: float
    lr_decay: DecayFunction
    memory_size: int
    learn_per_step: int
    steps_btw_train: int
    nb_samples: int
    beta: Optional[float]

class ApproximateAdvantagePolicyConfig(NamedTuple):
    batch_size: int
    gamma: float
    dt: float
    lr: float
    policy_lr: float
    lr_decay: DecayFunction
    memory_size: int
    learn_per_step: int
    steps_btw_train: int
    beta: Optional[float]


class EnvConfig(NamedTuple):
    id: str
    dt: float
    time_limit: Optional[float]


NoiseConfig = Union[ParameterNoiseConfig, ActionNoiseConfig]
PolicyConfig = Union[
    AdvantagePolicyConfig,
    SampledAdvantagePolicyConfig,
    ApproximateAdvantagePolicyConfig
]

def read_config(
        args,
        config_file: Optional[str]=None) -> Tuple[
            PolicyConfig, NoiseConfig, NoiseConfig, EnvConfig]:
    # No configuration file at the moment
    if args.cyclic_exploration:
        def noise_decay(t: int) -> float:
            return np.sin(t * args.dt) ** 2
    else:
        def noise_decay(_: int) -> float: # type: ignore
            return 1

    def lr_decay(t):
        return 1 / np.power(1 + args.dt * t, 0)

    policy_config_dict = dict(
        batch_size=args.batch_size, gamma=args.gamma, dt=args.dt,
        lr=args.lr, lr_decay=lr_decay,
        memory_size=args.memory_size, learn_per_step=args.learn_per_step,
        steps_btw_train=args.steps_btw_train, beta=args.beta)
    if args.policy_lr is not None:
        policy_config_dict['policy_lr'] = args.policy_lr
        policy_config: PolicyConfig = ApproximateAdvantagePolicyConfig(
            **policy_config_dict)
    elif args.nb_policy_samples is not None:
        policy_config_dict['nb_samples'] = args.nb_policy_samples
        policy_config = SampledAdvantagePolicyConfig(
            **policy_config_dict)
    else:
        policy_config = AdvantagePolicyConfig(**policy_config_dict)
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

    env_config = EnvConfig(args.env_id, args.dt, args.time_limit)
    return policy_config, noise_config, eval_noise_config, env_config
