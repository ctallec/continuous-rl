""" Trying to solve pendulum using a robust advantage learning. """
# pylint: disable=too-many-arguments, too-many-locals
from functools import partial
import argparse
import numpy as np
import torch

from abstract import Policy, Env, Arrayable
from config import NoiseConfig, PolicyConfig, EnvConfig, read_config
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
        batch_size: int,
        hidden_size: int,
        nb_layers: int,
        nb_epochs: int,
        nb_steps: int,
        nb_eval_env: int,
        policy_config: PolicyConfig,
        noise_config: NoiseConfig,
        eval_noise_config: NoiseConfig,
        env_config: EnvConfig):
    """ Starts training. """
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setting up envs
    env_fn = partial(make_env, env_config)
    env: Env = SubprocVecEnv([env_fn() for _ in range(batch_size)]) # type: ignore
    eval_env: Env = SubprocVecEnv([env_fn() for _ in range(nb_eval_env)]) # type: ignore
    obs = env.reset()

    policy, eval_policy = \
        setup_policy(env.observation_space, env.action_space, policy_config,
                     nb_layers, batch_size, hidden_size, noise_config,
                     eval_noise_config, device)

    for e in range(nb_epochs):
        print(f"Epoch {e}...")
        obs = train(nb_steps, env, policy, obs)
        evaluate(env_config.dt, e, eval_env, eval_policy)
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
    parser.add_argument('--alpha', type=float, default=3.)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--nb_epochs', type=int, default=50000)
    parser.add_argument('--nb_steps', type=int, default=500)
    parser.add_argument('--sigma_eval', type=float, default=0)
    parser.add_argument('--sigma', type=float, default=.3)
    parser.add_argument('--theta', type=float, default=1)
    parser.add_argument('--nb_eval_env', type=int, default=100)
    parser.add_argument('--memory_size', type=int, default=10000)
    parser.add_argument('--learn_per_step', type=int, default=3)
    parser.add_argument('--cyclic_exploration', action='store_true')
    parser.add_argument('--lr', type=float, default=.003)
    args = parser.parse_args()
    policy_config, noise_config, eval_noise_config, env_config = read_config(args)
    main(batch_size=args.batch_size,
         hidden_size=args.hidden_size, nb_layers=args.nb_layers,
         nb_epochs=args.nb_epochs, nb_steps=args.nb_steps,
         nb_eval_env=args.nb_eval_env, policy_config=policy_config,
         noise_config=noise_config, eval_noise_config=eval_noise_config,
         env_config=env_config)
