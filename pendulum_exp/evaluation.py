"""Environment specific evaluation."""
import numpy as np
import matplotlib.pyplot as plt
from abstract import Policy, Env
from policy import AdvantagePolicy
from envs.vecenv import SubprocVecEnv
from envs.pusher import PusherEnv
from gym.envs.classic_control import PendulumEnv

def specific_evaluation(
        epoch: int,
        log: int,
        dt: float,
        env: Env,
        policy: Policy):
    assert isinstance(policy, AdvantagePolicy), f"Incorrect policy type: {type(policy)}, "\
        "AdvantagePolicy expected."
    assert isinstance(env, SubprocVecEnv), f"Incorrect environment type: {type(env)}, "\
        "SubprocVecEnv expected."

    if isinstance(env.envs[0], PusherEnv):
        nb_pixels = 50
        state_space = np.linspace(-1.5, 1.5, nb_pixels)[:, np.newaxis]

        vs = policy.value(state_space)
        plt.clf()
        plt.plot(state_space, vs)
        plt.pause(.1)
    elif isinstance(env.envs[0], PendulumEnv):
        nb_pixels = 50
        theta_space = np.linspace(-np.pi, np.pi, nb_pixels)
        dtheta_space = np.linspace(-10, 10, nb_pixels)
        theta, dtheta = np.meshgrid(theta_space, dtheta_space)
        state_space = np.stack([np.cos(theta), np.sin(theta), dtheta], axis=-1)

        vs = policy.value(state_space).squeeze()
        actions = policy.step(state_space)
        plt.clf()
        plt.subplot(121)
        plt.imshow(actions, origin='lower')
        plt.subplot(122)
        plt.imshow(vs, origin='lower')
        plt.colorbar()
        plt.pause(.1)
        if epoch % log == log - 1:
            plt.savefig(f'logs/results_{dt}_{epoch}.pdf')
