"""Environment specific evaluation."""
import numpy as np
import matplotlib.pyplot as plt
from envs.env import Env
from agents.agent import Agent
from agents.off_policy.offline_agent import OfflineAgent
from envs.hill import HillEnv
from envs.pusher import AbstractPusher, ContinuousPusherEnv
from convert import th_to_arr
from gym.envs.classic_control import PendulumEnv
from gym.spaces import Box
from mylog import log_image


def specific_evaluation(
        epoch: int,
        log: int,
        dt: float,
        env: Env,
        agent: Agent):

    if isinstance(env.envs[0].unwrapped, AbstractPusher): # type: ignore
        nb_pixels = 50
        state_space = np.linspace(-1.5, 1.5, nb_pixels)[:, np.newaxis]

        actions = agent.actions(state_space)
        values = agent.value(state_space)
        plt.clf()
        plt.subplot(131)
        plt.plot(state_space, th_to_arr(values))
        plt.subplot(132)
        plt.plot(state_space, th_to_arr(actions))
        if isinstance(env.envs[0].unwrapped, ContinuousPusherEnv): # type: ignore
            action_space = np.linspace(-1, 1, nb_pixels)[:, np.newaxis]
            states, actions = np.meshgrid(state_space, action_space)
            states = states[..., np.newaxis]
            actions = actions[..., np.newaxis]
            if isinstance(agent, OfflineAgent):
                advantage = th_to_arr(agent.advantage(states, actions).squeeze())
                plt.subplot(133)
                plt.imshow(advantage)
                log_image('adv', epoch, advantage)
        plt.pause(.1)
    elif isinstance(env.envs[0].unwrapped, PendulumEnv): # type: ignore
        nb_pixels = 50
        theta_space = np.linspace(-np.pi, np.pi, nb_pixels)
        dtheta_space = np.linspace(-10, 10, nb_pixels)
        theta, dtheta = np.meshgrid(theta_space, dtheta_space)
        state_space = np.stack([np.cos(theta), np.sin(theta), dtheta], axis=-1)
        target_shape = state_space.shape[:2]

        actions = agent.actions(
            state_space.reshape(-1, 3))
        values = agent.value(
            state_space.reshape(-1, 3)).reshape(target_shape).squeeze()
        if isinstance(agent, OfflineAgent):
            advs = agent.advantage(
                state_space.reshape(-1, 3),
                actions).reshape(target_shape).squeeze()
            non_advs = agent.advantage(
                state_space.reshape(-1, 3),
                1 - actions).reshape(target_shape).squeeze()

        actions = actions.reshape(target_shape).squeeze()
        plt.figure(0, figsize=(20, 10))
        plt.clf()
        plt.subplot(241)
        plt.imshow(th_to_arr(actions), origin='lower')
        log_image('act', epoch, th_to_arr(actions))
        plt.subplot(242)
        plt.imshow(th_to_arr(values), origin='lower')
        log_image('val', epoch, th_to_arr(values))
        if isinstance(agent, OfflineAgent):
            plt.subplot(243)
            plt.imshow(th_to_arr(advs), origin='lower')
            log_image('adv', epoch, th_to_arr(advs))
            plt.subplot(244)
            plt.imshow(th_to_arr(non_advs), origin='lower')
            log_image('inverse_adv', epoch, th_to_arr(non_advs))
            plt.subplot(245)
            plt.hist(th_to_arr(values).reshape(-1), bins=nb_pixels)
            plt.subplot(246)
            plt.hist(th_to_arr(non_advs).reshape(-1), bins=nb_pixels)
        plt.colorbar()
        plt.pause(.1)
    elif isinstance(env.envs[0].unwrapped, HillEnv):
        nb_pixels = 50
        state_space = np.linspace(-1, 1, nb_pixels)[:, np.newaxis]

        actions = agent.actions(state_space)
        values = agent.value(state_space).squeeze()

        plt.clf()
        plt.subplot(1, 3, 1)
        plt.plot(state_space, th_to_arr(values))
        plt.subplot(1, 3, 2)
        plt.plot(state_space, th_to_arr(actions))
        if isinstance(env.envs[0].unwrapped.action_space, Box): # type: ignore
            action_space = np.linspace(-1, 1, nb_pixels)[:, np.newaxis]
            states, actions = np.meshgrid(state_space, action_space)
            states = states[..., np.newaxis]
            actions = actions[..., np.newaxis]
            advantages = agent.advantage(states, actions).squeeze()
            plt.subplot(1, 3, 3)
            plt.imshow(th_to_arr(advantages))
            log_image('adv', epoch, th_to_arr(advantages))
        plt.pause(.1)
