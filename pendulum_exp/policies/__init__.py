"""Policies subpackage."""
from typing import Tuple, Type, Any
from gym import Space
from gym.spaces import Discrete, Box
from abstract import Policy
from policies.discrete import AdvantagePolicy
from policies.mixture import ContinuousAdvantageMixturePolicy
from policies.continuous import AdvantagePolicy as ContinuousAdvantagePolicy
from policies.continuous import SampledAdvantagePolicy
from policies.benchmarkspolicies.dqn import DQNPolicy
from models import ContinuousAdvantageMLP, ContinuousPolicyMLP, MLP, NormalizedMLP
from noise import setup_noise
from config import NoiseConfig, PolicyConfig
from config import SampledAdvantagePolicyConfig, ApproximateAdvantagePolicyConfig
from config import AdvantagePolicyConfig, DQNConfig

def setup_policy(observation_space: Space, # flake8: noqa
                 action_space: Space,
                 policy_config: PolicyConfig,
                 nb_layers: int,
                 nb_train_env: int,
                 nb_eval_env: int,
                 hidden_size: int,
                 normalize_state: bool,
                 noise_config: NoiseConfig,
                 eval_noise_config: NoiseConfig,
                 device) -> Tuple[Policy, Policy]:
    # assume observation space has shape (F,) for now
    nb_state_feats = observation_space.shape[0]

    if hasattr(policy_config, 'mixture') and policy_config.mixture: # type: ignore
        nb_outputs = 4
    else:
        nb_outputs = 1

    # normalization
    if normalize_state:
        def maker(maker_cls, **kwargs):
            model = maker_cls(**kwargs)
            return NormalizedMLP(model)
    else:
        def maker(maker_cls, **kwargs):
            return maker_cls(**kwargs)

    if isinstance(policy_config, (AdvantagePolicyConfig, SampledAdvantagePolicyConfig,
                  ApproximateAdvantagePolicyConfig)):
        net_dict = dict(nb_layers=nb_layers, hidden_size=hidden_size)

        val_function = maker(MLP, nb_inputs=nb_state_feats, nb_outputs=1, **net_dict).to(device)

        policy_dict = dict(val_function=val_function, policy_config=policy_config, device=device)

        if isinstance(policy_config, AdvantagePolicyConfig):
            assert isinstance(action_space, Discrete)
            adv_function = maker(MLP, nb_inputs=nb_state_feats, nb_outputs=action_space.n,
                                 **net_dict).to(device)
            noise = setup_noise(noise_config, network=adv_function,
                                action_shape=(nb_train_env, action_space.n)).to(device)
            eval_noise = setup_noise(eval_noise_config, network=adv_function,
                                     action_shape=(nb_eval_env, action_space.n)).to(device)

            policy_dict["adv_function"] = adv_function

            policy: Policy = AdvantagePolicy(adv_noise=noise, **policy_dict)
            eval_policy: Policy = AdvantagePolicy(adv_noise=eval_noise, **policy_dict)

        elif isinstance(policy_config, (ApproximateAdvantagePolicyConfig, SampledAdvantagePolicyConfig)):
            assert isinstance(action_space, Box)
            nb_actions = action_space.shape[-1]
            adv_function = maker(ContinuousAdvantageMLP, nb_outputs=nb_outputs,
                                 nb_state_feats=nb_state_feats, nb_actions=nb_actions,
                                 **net_dict).to(device)
            policy_dict["adv_function"] = adv_function

            if isinstance(policy_config, ApproximateAdvantagePolicyConfig):
                policy_function = maker(ContinuousPolicyMLP,
                                        nb_inputs=nb_state_feats, nb_outputs=nb_actions,
                                        **net_dict).to(device)
                policy_dict["policy_function"] = policy_function

                noise_dict = dict(network=policy_function,
                                  action_shape=(nb_train_env, action_space.shape[0]))
                noise = setup_noise(noise_config, **noise_dict).to(device)
                eval_noise = setup_noise(eval_noise_config, **noise_dict).to(device)

                if policy_config.mixture:
                    policy_cls: Type[Any] = ContinuousAdvantageMixturePolicy
                else:
                    policy_cls = ContinuousAdvantagePolicy
                policy = policy_cls(policy_noise=noise, **policy_dict)
                eval_policy = policy_cls(policy_noise=eval_noise, **policy_dict)

            elif isinstance(policy_config, SampledAdvantagePolicyConfig):
                noise_dict = dict(network=adv_function, action_shape=(nb_train_env, 1))
                noise = setup_noise(noise_config, **noise_dict).to(device)
                eval_noise = setup_noise(eval_noise_config, **noise_dict).to(device)
                policy_dict["action_shape"] = action_space.shape

                policy = SampledAdvantagePolicy(adv_noise=noise, **policy_dict)
                eval_policy = SampledAdvantagePolicy(adv_noise=eval_noise, **policy_dict)

    elif isinstance(policy_config, DQNConfig):
        qnet_function = maker(MLP, nb_inputs=nb_state_feats, nb_outputs=action_space.n,
                              nb_layers=nb_layers, hidden_size=hidden_size).to(device)
        noise = setup_noise(noise_config, network=qnet_function,
                            action_shape=(nb_train_env, action_space.n)).to(device)
        eval_noise = setup_noise(eval_noise_config, network=qnet_function,
                                 action_shape=(nb_eval_env, action_space.n)).to(device)
        policy = DQNPolicy(qnet_function=qnet_function, qnet_noise=noise,
                           policy_config=policy_config, device=device)
        eval_policy = DQNPolicy(qnet_function=qnet_function, qnet_noise=eval_noise,
                                policy_config=policy_config, device=device)

    else:
        raise NotImplementedError()

    return policy, eval_policy


__all__ = ['AdvantagePolicy', 'ContinuousAdvantagePolicy']
