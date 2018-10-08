""" Trying to solve pendulum using a robust advantage learning. """
# pylint: disable=too-many-arguments, too-many-locals
import argparse
from typing import Tuple
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as f

from envs.vecenv import SubprocVecEnv
from models import MLP, perturbed_output
from stats import FloatingAvg
from noise import ParameterNoise

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
        anoise: ParameterNoise,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.Optimizer],
        schedulers: torch.optim.lr_scheduler.LambdaLR,
        vec_env: SubprocVecEnv,
        start_obs: np.ndarray,
        dt: float,
        epsilon: float,
        gamma: float,
        reward_floating_mean: FloatingAvg,
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
            # pre_action = afunction(obs)
            pre_action = perturbed_output(obs, afunction, anoise)
            anoise.step()

            action = 4 * to_numpy(torch.max(pre_action, dim=1)[1]) - 2
            next_obs, reward, _, _ = vec_env.step(action[:, np.newaxis])
            next_obs = next_obs
            true_reward = reward * dt
            reward_floating_mean.step(true_reward)
            true_reward = true_reward - reward_floating_mean.mean

        # learn
        next_obs = to_tensor(next_obs, device)
        v = vfunction(obs).squeeze()
        expected_v = to_tensor(true_reward, device) + \
            gamma ** dt *  vfunction(next_obs).detach().squeeze()
        v_loss = f.mse_loss(v, expected_v)

        indices = torch.LongTensor(
            (action == 2.0).astype('int64'))
        a = afunction(obs).gather(1, indices.view(-1, 1)).squeeze()
        a_loss = f.mse_loss(a, (expected_v - v.detach()) / dt)

        optimizers[0].zero_grad()
        v_loss.backward()
        optimizers[0].step()
        optimizers[1].zero_grad()
        a_loss.backward()
        optimizers[1].step()
        schedulers[0].step()
        schedulers[1].step()

        cum_aloss += a_loss.item()
        cum_vloss += v_loss.item()

        obs = next_obs
    print(f'At epoch {epoch}, '
          f'avg_aloss: {cum_aloss / nsteps}, '
          f'avg_vloss: {cum_vloss / nsteps}')
    return to_numpy(next_obs)

def evaluate(
        epoch: int,
        vfunction: MLP,
        afunction: MLP,
        device: torch.device):
    """ Evaluate. """
    log = 10

    if epoch % log == log - 1:
        env = gym.make('Pendulum-v0').unwrapped
        nb_steps = 250
        with torch.no_grad():
            obs = to_tensor(env.reset(), device).unsqueeze(0)
            for _ in range(nb_steps):
                action = 4 * torch.max(afunction(obs), dim=1)[1][0] - 2
                obs, _, _, _ = env.step(action)
                obs = to_tensor(obs, device).unsqueeze(0)
                env.render()
        env.close()

    with torch.no_grad():
        vfunction.eval()
        nb_pixels = 50
        theta_space = np.linspace(-np.pi, np.pi, nb_pixels)
        dtheta_space = np.linspace(-3, 3, nb_pixels)
        theta, dtheta = np.meshgrid(theta_space, dtheta_space)
        state_space = np.stack([np.cos(theta), np.sin(theta), dtheta], axis=-1)

        vs = to_numpy(vfunction(to_tensor(state_space, device)))
        plt.clf()
        plt.imshow(vs.squeeze())
        plt.colorbar()
        plt.pause(.1)


def main(
        batch_size: int,
        hidden_size: int,
        nb_layers: int,
        nb_epochs: int,
        nb_steps: int,
        sigma: float,
        theta: float,
        lr: float,
        epsilon: float,
        gamma: float):
    """ Starts training. """
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setting up envs
    # TODO: check that reward is scaled properly (by dt)
    envs = [gym.make('Pendulum-v0') for _ in range(batch_size)]
    envs = [env.unwrapped for env in envs]
    dt = envs[0].dt
    nb_inputs = 3
    nb_actions = 2
    vec_env = SubprocVecEnv(envs)
    obs = vec_env.reset()

    def lr_decay(t):
        return 1 # / np.log(2 + t * dt)

    reward_floating_mean = FloatingAvg(
        dt
    )

    # setting up models
    vfunction = MLP(nb_inputs=nb_inputs, nb_outputs=1,
                    nb_layers=nb_layers, hidden_size=hidden_size).to(device)
    afunction = MLP(nb_inputs=nb_inputs, nb_outputs=nb_actions,
                    nb_layers=nb_layers, hidden_size=hidden_size).to(device)
    a_noise = ParameterNoise(afunction, theta, sigma, dt)

    # TODO: implement learning rate decay
    optimizers = (
        torch.optim.SGD(vfunction.parameters(), lr=lr),
        torch.optim.SGD(afunction.parameters(), lr=lr * dt))
    schedulers = (
        torch.optim.lr_scheduler.LambdaLR(optimizers[0], lr_decay),
        torch.optim.lr_scheduler.LambdaLR(optimizers[1], lr_decay))

    for e in range(nb_epochs):
        print(f"Epoch {e}...")
        obs = train(e, vfunction, afunction, a_noise, optimizers, schedulers,
                    vec_env, obs, dt, epsilon, gamma,
                    reward_floating_mean, nb_steps, device)
        evaluate(e, vfunction, afunction, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default=.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--nb_layers', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--nb_epochs', type=int, default=5000)
    parser.add_argument('--nb_steps', type=int, default=500)
    parser.add_argument('--sigma', type=float, default=.4)
    parser.add_argument('--theta', type=float, default=1)
    parser.add_argument('--lr', type=float, default=.003)
    args = parser.parse_args()
    main(**vars(args))
