""" Trying to solve pendulum using a robust advantage learning. """
# pylint: disable=too-many-arguments, too-many-locals
from itertools import chain
import argparse
import gym
import torch
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt

from envs.vecenv import SubprocVecEnv
from models import MLP
from stats import FloatingAvg, epsilon_noise

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
        vfunction: MLP,
        afunction: MLP,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        vec_env: SubprocVecEnv,
        start_obs: np.ndarray,
        dt: float,
        epsilon: float,
        gamma: float,
        reward_floating_mean: FloatingAvg,
        nsteps: int,
        device: torch.device):
    """ Trains for one epoch. """
    # TODO: fix epsilon problem (what we want is an Ornstein Ulhenbeck on
    # the weights)
    vfunction.train()
    afunction.train()
    obs = to_tensor(start_obs, device)
    for _ in range(nsteps):
        # interact
        with torch.no_grad():
            action = 4 * to_numpy(torch.max(afunction(obs), dim=1)[1]) - 2
            action = epsilon_noise(action, epsilon)
            next_obs, reward, _, _ = vec_env.step(action[:, np.newaxis])
            next_obs = xy_to_theta(next_obs)
            true_reward = reward * dt
            reward_floating_mean.step(true_reward)
            true_reward = true_reward - reward_floating_mean.mean

        # learn
        next_obs = to_tensor(next_obs, device)
        v = vfunction(obs).squeeze()
        expected_v = to_tensor(true_reward, device) + gamma * vfunction(next_obs).detach().squeeze()
        v_loss = f.mse_loss(v, expected_v)

        indices = torch.LongTensor(
            (action == 2.0).astype('int64'))
        a = afunction(obs).gather(1, indices.view(-1, 1)).squeeze()
        a_loss = f.mse_loss(a, expected_v - v.detach())

        optimizer.zero_grad()
        (v_loss + a_loss).backward()
        optimizer.step()
        scheduler.step()

        obs = next_obs
    return to_numpy(next_obs)

def evaluate(
        vfunction: MLP,
        device: torch.device):
    """ Evaluate. """
    with torch.no_grad():
        vfunction.eval()
        nb_pixels = 50
        theta_space = np.linspace(-np.pi, np.pi, nb_pixels)
        dtheta_space = np.linspace(-5, 5, nb_pixels)
        state_space = np.stack(np.meshgrid(theta_space, dtheta_space), axis=-1)

        vs = to_numpy(vfunction(to_tensor(state_space, device)))
        plt.imshow(vs.squeeze())
        plt.pause(.1)


def main(
        batch_size: int,
        hidden_size: int,
        nb_layers: int,
        nb_epochs: int,
        nb_steps: int,
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
    nb_inputs = 2
    nb_actions = 2
    vec_env = SubprocVecEnv(envs)
    obs = xy_to_theta(vec_env.reset())
    reward_floating_mean = FloatingAvg(lr * dt)

    # setting up models
    vfunction = MLP(nb_inputs=nb_inputs, nb_outputs=1,
                    nb_layers=nb_layers, hidden_size=hidden_size).to(device)
    afunction = MLP(nb_inputs=nb_inputs, nb_outputs=nb_actions,
                    nb_layers=nb_layers, hidden_size=hidden_size).to(device)

    # TODO: implement learning rate decay
    optimizer = torch.optim.SGD(chain(vfunction.parameters(), afunction.parameters()),
                                lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda t: 1 / np.power(1 + dt * t, 3/5))

    for e in range(nb_epochs):
        print(f"Epoch {e}...")
        obs = train(vfunction, afunction, optimizer, scheduler,
                    vec_env, obs, dt, epsilon, gamma,
                    reward_floating_mean, nb_steps, device)
        evaluate(vfunction, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default=.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--nb_layers', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--nb_epochs', type=int, default=500)
    parser.add_argument('--nb_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=.1)
    args = parser.parse_args()
    main(**vars(args))
