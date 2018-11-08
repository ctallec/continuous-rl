"""Policies subpackage."""
from typing import Tuple
from gym import Space
from gym.spaces import Discrete, Box
from abstract import Policy
from policies.discrete import AdvantagePolicy
from policies.continuous import AdvantagePolicy as ContinuousAdvantagePolicy
from policies.continuous import SampledAdvantagePolicy
from policies.wrappers import StateNormalization
from models import ContinuousAdvantageMLP, ContinuousPolicyMLP, MLP
from noise import setup_noise
from config import NoiseConfig, PolicyConfig, AdvantagePolicyConfig

def setup_policy(observation_space: Space,
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
    val_function = MLP(nb_inputs=nb_state_feats, nb_outputs=1,
                       nb_layers=nb_layers, hidden_size=hidden_size).to(device)
    if normalize_state:
        normalization_state = { # we need to share normalization between policies
            'mean': None,
            'mean_squares': None
        }
    if isinstance(action_space, Discrete):
        assert isinstance(policy_config, AdvantagePolicyConfig)
        adv_function = MLP(nb_inputs=nb_state_feats, nb_outputs=action_space.n,
                           nb_layers=nb_layers, hidden_size=hidden_size).to(device)
        noise = setup_noise(noise_config, network=adv_function,
                            action_shape=(nb_train_env, action_space.n)).to(device)
        eval_noise = setup_noise(eval_noise_config, network=adv_function,
                                 action_shape=(nb_eval_env, action_space.n)).to(device)
        adv_function = MLP(nb_inputs=nb_state_feats, nb_outputs=action_space.n,
                           nb_layers=nb_layers, hidden_size=hidden_size).to(device)
        noise = setup_noise(noise_config, network=adv_function,
                            action_shape=(nb_train_env, action_space.n)).to(device)
        eval_noise = setup_noise(eval_noise_config, network=adv_function,
                                 action_shape=(nb_eval_env, action_space.n)).to(device)
        policy: Policy = AdvantagePolicy(
            adv_function=adv_function, val_function=val_function, adv_noise=noise,
            policy_config=policy_config, device=device)
        eval_policy: Policy = AdvantagePolicy(
            adv_function=adv_function, val_function=val_function, adv_noise=eval_noise,
            policy_config=policy_config, device=device)

    elif isinstance(action_space, Box):
        nb_actions = action_space.shape[-1]
        adv_function = ContinuousAdvantageMLP(
            nb_state_feats=nb_state_feats, nb_actions=nb_actions,
            nb_layers=nb_layers, hidden_size=hidden_size).to(device)
        if isinstance(policy_config, AdvantagePolicyConfig):
            policy_function = ContinuousPolicyMLP(
                nb_inputs=nb_state_feats, nb_outputs=nb_actions,
                nb_layers=nb_layers, hidden_size=hidden_size).to(device)
            noise = setup_noise(noise_config, network=policy_function,
                                action_shape=(nb_train_env, action_space.shape[0])).to(device)
            eval_noise = setup_noise(eval_noise_config, network=policy_function,
                                     action_shape=(nb_eval_env, action_space.shape[0])).to(device)
            policy = ContinuousAdvantagePolicy(
                adv_function=adv_function, val_function=val_function, policy_function=policy_function,
                policy_noise=noise, policy_config=policy_config, device=device)
            eval_policy = ContinuousAdvantagePolicy(
                adv_function=adv_function, val_function=val_function, policy_function=policy_function,
                policy_noise=eval_noise, policy_config=policy_config, device=device)
        else:
            noise = setup_noise(noise_config, network=adv_function,
                                action_shape=(nb_train_env, 1)).to(device)
            eval_noise = setup_noise(eval_noise_config, network=adv_function,
                                     action_shape=(nb_eval_env, 1)).to(device)
            policy = SampledAdvantagePolicy(
                adv_function=adv_function, val_function=val_function,
                adv_noise=noise, policy_config=policy_config, action_shape=action_space.shape,
                device=device)
            eval_policy = SampledAdvantagePolicy(
                adv_function=adv_function, val_function=val_function,
                adv_noise=eval_noise, policy_config=policy_config, action_shape=action_space.shape,
                device=device)

    if normalize_state:
        policy = StateNormalization(policy, normalization_state)
        eval_policy = StateNormalization(eval_policy, normalization_state)
    return policy, eval_policy


__all__ = ['AdvantagePolicy', 'ContinuousAdvantagePolicy']
