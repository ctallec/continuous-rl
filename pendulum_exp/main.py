""" Trying to solve pendulum using a robust advantage learning. """
# pylint: disable=too-many-arguments, too-many-locals
from functools import partial
import argparse
import numpy as np
import torch

from abstract import Policy, Env, Arrayable
from config import NoiseConfig, ActionNoiseConfig, ParameterNoiseConfig
from policies import setup_policy
from interact import interact
from envs.vecenv import SubprocVecEnv
from envs.utils import make_env
from evaluation import specific_evaluation
from utils import compute_return

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
    log = int(.1 / dt)
    video_log = 10
    policy.eval()

    if epoch % log == log - 1:
        rewards, dones = [], []
        imgs = []
        nb_steps = int(10 / dt)
        obs = env.reset()
        for _ in range(nb_steps):
            obs, reward, done = interact(env, policy, obs)
            rewards.append(reward)
            dones.append(done)
            if (epoch // log) % video_log == video_log - 1:
                imgs.append(env.render())
        R = compute_return(np.stack(rewards, axis=0),
                           np.stack(dones, axis=0))
        print(f"At epoch {epoch}, return: {R}")

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
        'pusher': (1, 3),
        'cartpole': (4, 2)
    }[env_id]
    env_fn = partial(make_env, env_id=env_id, dt=dt)
    env: Env = SubprocVecEnv([env_fn() for _ in range(batch_size)]) # type: ignore
    eval_env: Env = SubprocVecEnv([env_fn() for _ in range(nb_eval_env)]) # type: ignore
    obs = env.reset()

    def lr_decay(t):
        return 1 / np.power(1 + dt * t, 1/4)

    def noise_decay(_):
        return 1

    if noise_type == 'parameter':
        noise_config: NoiseConfig = ParameterNoiseConfig(
            sigma, theta, dt, noise_decay)
        eval_noise_config: NoiseConfig = ParameterNoiseConfig(
            sigma_eval, theta, dt, noise_decay)
    else:
        noise_config = ActionNoiseConfig(
            sigma, theta, dt, noise_decay)
        eval_noise_config = ActionNoiseConfig(
            sigma_eval, theta, dt, noise_decay)

    policy, eval_policy = \
        setup_policy(env.observation_space, env.action_space, gamma, lr, dt,
                     lr_decay, nb_layers, batch_size, hidden_size, noise_config,
                     eval_noise_config, device)

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
