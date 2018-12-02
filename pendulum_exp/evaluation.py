"""Environment specific evaluation."""
import numpy as np
import matplotlib.pyplot as plt
from abstract import Policy, Env
from policies import ContinuousAdvantagePolicy
from policies.mixture import ContinuousAdvantageMixturePolicy
from envs.vecenv import SubprocVecEnv
from envs.hill import HillEnv
from envs.pusher import AbstractPusher, ContinuousPusherEnv
from convert import th_to_arr
from gym.envs.classic_control import PendulumEnv
from gym.spaces import Box

def specific_evaluation(
        epoch: int,
        log: int,
        dt: float,
        env: Env,
        policy: Policy):
    assert isinstance(env, SubprocVecEnv), f"Incorrect environment type: {type(env)}, SubprocVecEnv expected." # type: ignore

    if isinstance(env.envs[0].unwrapped, AbstractPusher): # type: ignore
        nb_pixels = 50
        state_space = np.linspace(-1.5, 1.5, nb_pixels)[:, np.newaxis]

        vs = policy.value(state_space)
        actions = policy.step(state_space).squeeze()
        plt.clf()
        plt.subplot(131)
        plt.plot(state_space, vs)
        plt.subplot(132)
        plt.plot(state_space, actions)
        if isinstance(env.envs[0].unwrapped, ContinuousPusherEnv): # type: ignore
            assert isinstance(policy, ContinuousAdvantagePolicy)
            action_space = np.linspace(-1, 1, nb_pixels)[:, np.newaxis]
            states, actions = np.meshgrid(state_space, action_space)
            states = states[..., np.newaxis]
            actions = actions[..., np.newaxis]
            adv = policy.advantage(states, actions).squeeze()
            plt.subplot(133)
            plt.imshow(adv)
        plt.pause(.1)
    elif isinstance(env.envs[0].unwrapped, PendulumEnv): # type: ignore
        nb_pixels = 50
        theta_space = np.linspace(-np.pi, np.pi, nb_pixels)
        dtheta_space = np.linspace(-10, 10, nb_pixels)
        theta, dtheta = np.meshgrid(theta_space, dtheta_space)
        state_space = np.stack([np.cos(theta), np.sin(theta), dtheta], axis=-1)
        target_shape = state_space.shape[:2]

        vs = policy.value(state_space.reshape(-1, 3)).reshape(target_shape).squeeze()
        actions = policy.step(state_space.reshape(-1, 3)).reshape(target_shape).squeeze()
        plt.clf()
        plt.subplot(121)
        plt.imshow(actions, origin='lower')
        plt.subplot(122)
        plt.imshow(vs, origin='lower')
        plt.colorbar()
        plt.pause(.1)
    elif isinstance(env.envs[0].unwrapped, HillEnv):
        nb_pixels = 50
        state_space = np.linspace(-1, 1, nb_pixels)[:, np.newaxis]

        mixture = isinstance(policy, ContinuousAdvantageMixturePolicy)
        if mixture:
            nb_plots = 4
        else:
            nb_plots = 3

        vs = policy.value(state_space)
        actions = policy.step(state_space).squeeze()

        plt.clf()
        plt.subplot(1, nb_plots, 1)
        plt.plot(state_space, vs)
        plt.subplot(1, nb_plots, 2)
        plt.plot(state_space, actions)
        if isinstance(env.envs[0].unwrapped.action_space, Box): # type: ignore
            action_space = np.linspace(-1, 1, nb_pixels)[:, np.newaxis]
            states, actions = np.meshgrid(state_space, action_space)
            states = states[..., np.newaxis]
            actions = actions[..., np.newaxis]
            adv = policy.advantage(states, actions).squeeze()
            if mixture:
                _, _, logpi, _, _ = \
                    policy.compute_mixture_advantages(states, actions)
                pi = th_to_arr(logpi.exp())[..., 0]
                plt.subplot(1, nb_plots, nb_plots)
                plt.imshow(pi)
            plt.subplot(1, nb_plots, 3)
            plt.imshow(adv)
        plt.pause(.1)
