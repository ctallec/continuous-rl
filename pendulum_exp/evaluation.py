"""Environment specific evaluation."""
import numpy as np
import matplotlib.pyplot as plt
from abstract import Policy, Env
from policies import AdvantagePolicy, ContinuousAdvantagePolicy
from policies.wrappers import StateNormalization
from envs.vecenv import SubprocVecEnv
from envs.pusher import AbstractPusher, ContinuousPusherEnv
from envs.utils import WrapPendulum, WrapContinuousPendulum

def specific_evaluation(
        epoch: int,
        log: int,
        dt: float,
        env: Env,
        policy: Policy):
    assert isinstance(policy, (AdvantagePolicy, ContinuousAdvantagePolicy, StateNormalization)), f"Incorrect policy type: {type(policy)}, "\
        "AdvantagePolicy expected."
    assert isinstance(env.unwrapped, SubprocVecEnv), f"Incorrect environment type: {type(env)}, SubprocVecEnv expected." # type: ignore

    if isinstance(env.unwrapped.envs[0], AbstractPusher): # type: ignore
        nb_pixels = 50
        state_space = np.linspace(-1.5, 1.5, nb_pixels)[:, np.newaxis]

        vs = policy.value(state_space)
        actions = policy.step(state_space).squeeze()
        plt.clf()
        plt.subplot(131)
        plt.plot(state_space, vs)
        plt.subplot(132)
        plt.plot(state_space, actions)
        if isinstance(env.unwrapped.envs[0], ContinuousPusherEnv): # type: ignore
            assert isinstance(policy, ContinuousAdvantagePolicy)
            action_space = np.linspace(-1, 1, nb_pixels)[:, np.newaxis]
            states, actions = np.meshgrid(state_space, action_space)
            states = states[..., np.newaxis]
            actions = actions[..., np.newaxis]
            adv = policy.advantage(states, actions).squeeze()
            plt.subplot(133)
            plt.imshow(adv)
        plt.pause(.1)
    elif isinstance(env.unwrapped.envs[0], (WrapPendulum, WrapContinuousPendulum)): # type: ignore
        nb_pixels = 50
        theta_space = np.linspace(-np.pi, np.pi, nb_pixels)
        dtheta_space = np.linspace(-10, 10, nb_pixels)
        theta, dtheta = np.meshgrid(theta_space, dtheta_space)
        state_space = np.stack([np.cos(theta), np.sin(theta), dtheta], axis=-1)

        vs = policy.value(state_space).squeeze()
        actions = policy.step(state_space).squeeze()
        plt.clf()
        plt.subplot(121)
        plt.imshow(actions, origin='lower')
        plt.subplot(122)
        plt.imshow(vs, origin='lower')
        plt.colorbar()
        plt.pause(.1)
        if epoch % log == log - 1:
            plt.savefig(f'logs/results_{dt}_{epoch}.pdf')
