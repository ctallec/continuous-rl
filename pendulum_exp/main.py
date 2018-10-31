"""Trying to solve pendulum using a robust advantage learning."""
import sys
from os.path import join, exists
from logging import info, basicConfig, INFO
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
from mylog import log, logto

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
    log_gap = int(.1 / dt)
    video_log = 10
    policy.eval()

    R = None
    if epoch % log_gap == log_gap - 1:
        rewards, dones = [], []
        imgs = []
        nb_steps = int(10 / dt)
        obs = env.reset()
        for _ in range(nb_steps):
            obs, reward, done = interact(env, policy, obs)
            rewards.append(reward)
            dones.append(done)
            if (epoch // log_gap) % video_log == video_log - 1:
                imgs.append(env.render())
        R = compute_return(np.stack(rewards, axis=0),
                           np.stack(dones, axis=0))
        info(f"At epoch {epoch}, return: {R}")
        log("Return", R, epoch)

    specific_evaluation(epoch, log_gap, dt, env, policy)
    return R

def main(
        logdir: str,
        batch_size: int,
        hidden_size: int,
        nb_layers: int,
        nb_epochs: int,
        nb_steps: int,
        nb_eval_env: int,
        normalize_state: bool,
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
        setup_policy(observation_space=env.observation_space,
                     action_space=env.action_space, policy_config=policy_config,
                     nb_layers=nb_layers, batch_size=batch_size,
                     nb_eval_env=nb_eval_env, hidden_size=hidden_size,
                     noise_config=noise_config, eval_noise_config=eval_noise_config,
                     normalize_state=normalize_state, device=device)

    # load checkpoints if directory is not empty
    policy_file = join(args.logdir, 'best_policy.pt')
    R = - np.inf
    if exists(policy_file):
        state_dict = torch.load(policy_file)
        R = state_dict["return"]
        info(f"Loading policy with return {R}...")
        policy.load_state_dict(torch.load(policy_file))

    for e in range(nb_epochs):
        info(f"Epoch {e}...")
        obs = train(nb_steps, env, policy, obs)
        new_R = evaluate(env_config.dt, e, eval_env, eval_policy)
        if new_R is not None and new_R > R:
            info(f"Saving new policy with return {new_R}")
            state_dict = policy.state_dict()
            state_dict["return"] = new_R
            torch.save(state_dict, policy_file)
            R = new_R
    env.close()
    eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
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
    parser.add_argument('--normalize_state', action='store_true')
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--redirect_stdout', action='store_true')
    args = parser.parse_args()

    # configure logging
    if args.redirect_stdout:
        basicConfig(filename=join(args.logdir, 'out.log'), level=INFO)
    else:
        basicConfig(stream=sys.stdout, level=INFO)

    logto(join(args.logdir, 'logs.pkl'))

    policy_config, noise_config, eval_noise_config, env_config = read_config(args)
    main(
        logdir=args.logdir, batch_size=args.batch_size,
        hidden_size=args.hidden_size, nb_layers=args.nb_layers,
        nb_epochs=args.nb_epochs, nb_steps=args.nb_steps,
        nb_eval_env=args.nb_eval_env, policy_config=policy_config,
        noise_config=noise_config, eval_noise_config=eval_noise_config,
        env_config=env_config, normalize_state=args.normalize_state)
