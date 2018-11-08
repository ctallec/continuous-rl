"""Trying to solve pendulum using a robust advantage learning."""
from typing import Optional
import sys
from os.path import join, exists
from logging import info, basicConfig, INFO
from functools import partial
import argparse
import argload
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
from mylog import log, logto, log_video

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
        policy: Policy,
        time_limit: Optional[float]=None):
    """ Evaluate. """
    log_gap = int(.1 / dt)
    video_log = 10
    policy.eval()

    R = None
    if epoch % log_gap == log_gap - 1:
        rewards, dones = [], []
        imgs = []
        time_limit = time_limit if time_limit else 10
        nb_steps = int(time_limit / dt)
        obs = env.reset()
        for _ in range(nb_steps):
            obs, reward, done = interact(env, policy, obs)
            rewards.append(reward)
            dones.append(done)
            if (epoch // log_gap) % video_log == video_log - 1:
                imgs.append(env.render(mode='rgb_array'))
        R = compute_return(np.stack(rewards, axis=0),
                           np.stack(dones, axis=0))
        info(f"At epoch {epoch}, return: {R}")
        log("Return", R, epoch)
        if (epoch // log_gap) % video_log == video_log - 1:
            log_video("demo", epoch, np.stack(imgs, axis=0))

    specific_evaluation(epoch, log_gap, dt, env, policy)
    return R

def main(
        logdir: str,
        noreload: bool,
        nb_train_env: int,
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
    env: Env = SubprocVecEnv([env_fn() for _ in range(nb_train_env)]) # type: ignore
    eval_env: Env = SubprocVecEnv([env_fn() for _ in range(nb_eval_env)]) # type: ignore
    obs = env.reset()

    policy, eval_policy = \
        setup_policy(observation_space=env.observation_space,
                     action_space=env.action_space, policy_config=policy_config,
                     nb_layers=nb_layers, nb_train_env=nb_train_env,
                     nb_eval_env=nb_eval_env, hidden_size=hidden_size,
                     noise_config=noise_config, eval_noise_config=eval_noise_config,
                     normalize_state=normalize_state, device=device)

    # load checkpoints if directory is not empty
    policy_file = join(logdir, 'best_policy.pt')
    R = - np.inf
    if exists(policy_file) and not noreload:
        state_dict = torch.load(policy_file)
        R = state_dict["return"]
        info(f"Loading policy with return {R}...")
        policy.load_state_dict(torch.load(policy_file))

    for e in range(nb_epochs):
        info(f"Epoch {e}...")
        obs = train(nb_steps, env, policy, obs)
        new_R = evaluate(env_config.dt, e, eval_env, eval_policy, env_config.time_limit)
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
    parser.add_argument('--dt', type=float, default=.05)
    parser.add_argument('--steps_btw_train', type=int, default=10)
    parser.add_argument('--env_id', type=str, default='pendulum')
    parser.add_argument('--noise_type', type=str, default='parameter')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--nb_layers', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--nb_epochs', type=int, default=50000)
    parser.add_argument('--nb_steps', type=int, default=100)
    parser.add_argument('--sigma_eval', type=float, default=0)
    parser.add_argument('--sigma', type=float, default=1.5)
    parser.add_argument('--theta', type=float, default=7.5)
    parser.add_argument('--nb_train_env', type=int, default=32)
    parser.add_argument('--nb_eval_env', type=int, default=16)
    parser.add_argument('--memory_size', type=int, default=10000)
    parser.add_argument('--learn_per_step', type=int, default=50)
    parser.add_argument('--cyclic_exploration', action='store_true')
    parser.add_argument('--normalize_state', action='store_true')
    parser.add_argument('--lr', type=float, default=.03)
    parser.add_argument('--time_limit', type=float, default=None)
    parser.add_argument('--redirect_stdout', action='store_true')
    parser.add_argument('--policy_type', type=str, default='default')
    parser.add_argument('--nb_policy_samples', type=int)
    parser.add_argument('--noreload', action='store_true')
    parser = argload.ArgumentLoader(parser, to_reload=[
        'dt', 'steps_btw_train', 'env_id', 'noise_type', 'batch_size',
        'hidden_size', 'nb_layers', 'alpha', 'gamma', 'nb_epochs', 'nb_steps',
        'sigma_eval', 'sigma', 'theta', 'nb_train_env', 'nb_eval_env', 'memory_size',
        'learn_per_step', 'cyclic_expliration', 'normalize_state', 'lr', 'time_limit',
        'policy_type', 'nb_policy_samples'
    ])
    args = parser.parse_args()

    # configure logging
    if args.redirect_stdout:
        basicConfig(filename=join(args.logdir, 'out.log'), level=INFO)
    else:
        basicConfig(stream=sys.stdout, level=INFO)

    logto(args.logdir)

    policy_config, noise_config, eval_noise_config, env_config = read_config(args)
    main(
        logdir=args.logdir, noreload=args.noreload, nb_train_env=args.nb_train_env,
        hidden_size=args.hidden_size, nb_layers=args.nb_layers,
        nb_epochs=args.nb_epochs, nb_steps=args.nb_steps,
        nb_eval_env=args.nb_eval_env, policy_config=policy_config,
        noise_config=noise_config, eval_noise_config=eval_noise_config,
        env_config=env_config, normalize_state=args.normalize_state)
