""" Trying to solve pendulum using a robust advantage learning. """
# pylint: disable=too-many-arguments, too-many-locals
from functools import partial
import argparse
import numpy as np
import torch

from abstract import Policy, Env, Arrayable, Noise
from policy import AdvantagePolicy
from interact import interact
from envs.vecenv import SubprocVecEnv
from envs.utils import make_env
from models import MLP
from noise import ParameterNoise, ActionNoise
from evaluation import specific_evaluation

def train(
        nb_steps: int,
        env: Env,
        policy: Policy,
        start_obs: Arrayable):
    """ Trains for one epoch. """
    policy.train()

    obs = start_obs
    for _ in range(nb_steps):
        # interact
        obs, _, _ = interact(env, policy, obs)
    return obs

def evaluate(
        dt: float,
        epoch: int,
        env: Env,
        policy: Policy):
    """ Evaluate. """
    log = int(1 / dt)
    policy.eval()

    if epoch % log == log - 1:
        imgs = []
        nb_steps = int(10 / dt)
        obs = env.reset()
        for _ in range(nb_steps):
            obs, reward, done = interact(env, policy, obs)
            imgs.append(env.render())

    specific_evaluation(epoch, log, dt, env, policy)

def main(
        env_id: str,
        dt: float,
        batch_size: int,
        hidden_size: int,
        nb_layers: int,
        nb_epochs: int,
        nb_steps: int,
        nb_eval_env: int,
        noise_type: str,
        sigma: float,
        sigma_eval: float,
        theta: float,
        lr: float,
        gamma: float,
        avg_alpha: float):
    """ Starts training. """
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setting up envs
    nb_inputs, nb_actions = {
        'pendulum': (3, 2),
        'pusher': (1, 3)
    }[env_id]
    env_fn = partial(make_env, env_id=env_id, dt=dt)
    env: Env = SubprocVecEnv([env_fn() for _ in range(batch_size)]) # type: ignore
    eval_env: Env = SubprocVecEnv([env_fn() for _ in range(nb_eval_env)]) # type: ignore
    obs = env.reset()

    def lr_decay(t):
        return 1 / np.power(1 + dt * t, 1/2)

    def noise_decay(_):
        return 1

    # setting up models
    val_function = MLP(nb_inputs=nb_inputs, nb_outputs=1,
                       nb_layers=nb_layers, hidden_size=hidden_size).to(device)
    adv_function = MLP(nb_inputs=nb_inputs, nb_outputs=nb_actions,
                       nb_layers=nb_layers, hidden_size=hidden_size).to(device)
    if noise_type == 'parameter':
        adv_noise: Noise = \
            ParameterNoise(adv_function, theta, sigma, dt, noise_decay)
        adv_noise_eval: Noise = \
            ParameterNoise(adv_function, theta, sigma_eval, dt, noise_decay)
    else:
        adv_noise = ActionNoise((batch_size, nb_actions), theta, sigma, dt, noise_decay)
        adv_noise_eval = ActionNoise((1, nb_actions), theta, sigma_eval, dt, noise_decay)

    policy = AdvantagePolicy(
        adv_function=adv_function, val_function=val_function, adv_noise=adv_noise,
        gamma=gamma, avg_alpha=avg_alpha, dt=dt, lr=lr, lr_decay=lr_decay, device=device)
    eval_policy = AdvantagePolicy(
        adv_function=adv_function, val_function=val_function, adv_noise=adv_noise_eval,
        gamma=gamma, avg_alpha=avg_alpha, dt=dt, lr=lr, lr_decay=lr_decay, device=device)


    for e in range(nb_epochs):
        print(f"Epoch {e}...")
        obs = train(nb_steps, env, policy, obs)
        evaluate(dt, e, eval_env, eval_policy)
    env.close()
    eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=float, default=.05)
    parser.add_argument('--env_id', type=str, default='pendulum')
    parser.add_argument('--noise_type', type=str, default='parameter')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--nb_layers', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--nb_epochs', type=int, default=50000)
    parser.add_argument('--nb_steps', type=int, default=500)
    parser.add_argument('--sigma_eval', type=float, default=0)
    parser.add_argument('--sigma', type=float, default=.3)
    parser.add_argument('--theta', type=float, default=1)
    parser.add_argument('--nb_eval_env', type=int, default=100)
    parser.add_argument('--avg_alpha', type=float, default=1.)
    parser.add_argument('--lr', type=float, default=.003)
    args = parser.parse_args()
    main(**vars(args))
