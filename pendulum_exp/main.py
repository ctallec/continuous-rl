""" Trying to solve pendulum using a robust advantage learning. """
# pylint: disable=too-many-arguments, too-many-locals
from functools import partial
from time import sleep
import argparse
from typing import Tuple, Union, Callable
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as f

from envs.vecenv import SubprocVecEnv
from envs.pusher import PusherEnv
from envs.utils import make_env
from models import MLP, perturbed_output
from stats import FloatingAvg
from noise import ParameterNoise, ActionNoise

def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """ Np to Th. """
    return torch.from_numpy(arr).float().to(device)

def to_numpy(arr: torch.Tensor) -> np.ndarray:
    """ Th to Np. """
    return arr.cpu().numpy()

def xy_to_theta(arr: np.ndarray) -> np.ndarray:
    """ Returns theta from x, y. """
    theta = np.arctan2(arr[:, 1], arr[:, 0])[:, np.newaxis]
    return np.concatenate((theta, arr[:, 2:]),
                          axis=-1)

def train(
        epoch: int,
        vfunction: MLP,
        afunction: MLP,
        anoise: Union[ParameterNoise, ActionNoise],
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.Optimizer],
        schedulers: torch.optim.lr_scheduler.LambdaLR,
        vec_env: SubprocVecEnv,
        start_obs: np.ndarray,
        dt: float,
        gamma: float,
        value_floating_mean: FloatingAvg,
        nsteps: int,
        device: torch.device):
    """ Trains for one epoch. """
    vfunction.train()
    afunction.train()
    obs = to_tensor(start_obs, device)

    cum_aloss = 0
    cum_vloss = 0
    for _ in range(nsteps):
        # interact
        with torch.no_grad():
            # prepare action
            pre_action = perturbed_output(obs, afunction, anoise)
            anoise.step()

            action = to_numpy(torch.max(pre_action, dim=1)[1])
            true_action = to_numpy(torch.max(afunction(obs), dim=1)[1])

            next_obs, reward, _, _ = vec_env.step(action)
            next_obs = to_tensor(next_obs, device)
            true_reward = reward * dt


        if (action == true_action).any():
            # learn
            a_idx = torch.LongTensor(action)
            ta_idx = torch.LongTensor(true_action)
            on_policy_idx = a_idx == ta_idx

            v = vfunction(obs).squeeze()
            expected_v = to_tensor(true_reward, device) + \
                gamma ** dt * vfunction(next_obs).detach().squeeze()
            value_floating_mean.step(expected_v.detach())

            v_loss = f.mse_loss(v[on_policy_idx],
                                expected_v[on_policy_idx] - value_floating_mean.mean)
            optimizers[0].zero_grad()
            v_loss.backward()
            optimizers[0].step()
            schedulers[0].step()

            cum_vloss += v_loss.item()

            indices = torch.LongTensor(action).to(device)
            a = afunction(obs).gather(1, indices.view(-1, 1)).squeeze()
            a_loss = f.mse_loss(a, (expected_v - v.detach()) / dt)

            optimizers[1].zero_grad()
            a_loss.backward()
            optimizers[1].step()
            schedulers[1].step()

            cum_aloss += a_loss.item()

        obs = next_obs
    print(f'At epoch {epoch}, '
          f'avg_aloss: {cum_aloss / nsteps}, '
          f'avg_vloss: {cum_vloss / nsteps}')
    return to_numpy(next_obs)

def evaluate(
        env_fn: Callable[[], gym.Env],
        epoch: int,
        vfunction: MLP,
        afunction: MLP,
        anoise: Union[ParameterNoise, ActionNoise],
        device: torch.device):
    """ Evaluate. """
    log = 100

    env = env_fn()
    if epoch % log == log - 1:
        nb_steps = 500
        with torch.no_grad():
            obs = to_tensor(env.reset(), device).unsqueeze(0)
            for _ in range(nb_steps):
                pre_action = perturbed_output(obs, afunction, anoise)
                anoise.step()

                action = to_numpy(torch.max(pre_action, dim=1)[1][0])
                obs, _, _, _ = env.step(action)
                obs = to_tensor(obs, device).unsqueeze(0)
                env.render()
                sleep(.02)
        env.close()

    if isinstance(env, PusherEnv):
        with torch.no_grad():
            vfunction.eval()
            nb_pixels = 50
            state_space = np.linspace(-1.5, 1.5, nb_pixels)[:, np.newaxis]

            vs = to_numpy(vfunction(to_tensor(state_space, device)))
            plt.clf()
            plt.plot(state_space, vs)
            plt.pause(.1)

    else:
        with torch.no_grad():
            vfunction.eval()
            afunction.eval()
            nb_pixels = 50
            theta_space = np.linspace(-np.pi, np.pi, nb_pixels)
            dtheta_space = np.linspace(-10, 10, nb_pixels)
            theta, dtheta = np.meshgrid(theta_space, dtheta_space)
            state_space = np.stack([np.cos(theta), np.sin(theta), dtheta], axis=-1)

            vs = to_numpy(vfunction(
                to_tensor(state_space, device).view(-1, 3)).view(nb_pixels, nb_pixels))
            advs = to_numpy(torch.max(afunction(
                to_tensor(state_space, device).view(-1, 3)), dim=-1)[1].view(nb_pixels, nb_pixels))
            plt.clf()
            plt.subplot(121)
            plt.imshow(advs, origin='lower')
            plt.subplot(122)
            plt.imshow(vs, origin='lower')
            plt.colorbar()
            plt.pause(.1)


def main(
        env_id: str,
        dt: float,
        batch_size: int,
        hidden_size: int,
        nb_layers: int,
        nb_epochs: int,
        nb_steps: int,
        noise_type: str,
        sigma: float,
        sigma_eval: float,
        theta: float,
        lr: float,
        gamma: float):
    """ Starts training. """
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setting up envs
    nb_inputs, nb_actions = {
        'pendulum': (3, 2),
        'pusher': (1, 3)
    }[env_id]
    env_fn = partial(make_env, env_id=env_id, dt=dt)
    envs = [env_fn() for _ in range(batch_size)]
    vec_env = SubprocVecEnv(envs)
    obs = vec_env.reset()

    def lr_decay(_):
        return 1

    def noise_decay(_):
        return 1

    value_floating_mean = FloatingAvg(
        dt
    )

    # setting up models
    vfunction = MLP(nb_inputs=nb_inputs, nb_outputs=1,
                    nb_layers=nb_layers, hidden_size=hidden_size).to(device)
    afunction = MLP(nb_inputs=nb_inputs, nb_outputs=nb_actions,
                    nb_layers=nb_layers, hidden_size=hidden_size).to(device)
    if noise_type == 'parameter':
        a_noise: Union[ParameterNoise, ActionNoise] = \
            ParameterNoise(afunction, theta, sigma, dt, noise_decay)
        a_noise_eval: Union[ParameterNoise, ActionNoise] = \
            ParameterNoise(afunction, theta, sigma, dt, noise_decay)
    else:
        a_noise = ActionNoise((batch_size, nb_actions), theta, sigma, dt, noise_decay)
        a_noise_eval = ActionNoise((1, nb_actions), theta, sigma_eval, dt, noise_decay)

    optimizers = (
        torch.optim.SGD(vfunction.parameters(), lr=lr),
        torch.optim.SGD(afunction.parameters(), lr=lr * dt))
    schedulers = (
        torch.optim.lr_scheduler.LambdaLR(optimizers[0], lr_decay),
        torch.optim.lr_scheduler.LambdaLR(optimizers[1], lr_decay))

    for e in range(nb_epochs):
        print(f"Epoch {e}...")
        obs = train(e, vfunction, afunction, a_noise, optimizers, schedulers,
                    vec_env, obs, dt, gamma,
                    value_floating_mean, nb_steps, device)
        evaluate(env_fn, e, vfunction, afunction, a_noise_eval, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=float, default=.05)
    parser.add_argument('--env_id', type=str, default='pendulum')
    parser.add_argument('--noise_type', type=str, default='action')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--nb_layers', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--nb_epochs', type=int, default=5000)
    parser.add_argument('--nb_steps', type=int, default=500)
    parser.add_argument('--sigma_eval', type=float, default=0)
    parser.add_argument('--sigma', type=float, default=.3)
    parser.add_argument('--theta', type=float, default=1)
    parser.add_argument('--lr', type=float, default=.003)
    args = parser.parse_args()
    main(**vars(args))
