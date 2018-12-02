"""Evaluate a model."""
from os.path import join, exists
from functools import partial
from logging import info
import torch
import numpy as np

from abstract import Env
from main import evaluate
from config import PolicyConfig, NoiseConfig, EnvConfig
from parse import setup_args
from envs.utils import make_env
from envs.vecenv import SubprocVecEnv
from policies import setup_policy
from config import read_config
from mylog import logto


def main(logdir: str, hidden_size: int, nb_layers: int, nb_eval_env: int,
         policy_config: PolicyConfig, eval_noise_config: NoiseConfig, env_config: EnvConfig,
         normalize_state: bool):

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setting up envs
    env_fn = partial(make_env, env_config)
    eval_env: Env = SubprocVecEnv([env_fn() for _ in range(nb_eval_env)]) # type: ignore

    _, eval_policy = \
        setup_policy(observation_space=eval_env.observation_space,
                     action_space=eval_env.action_space, policy_config=policy_config,
                     nb_layers=nb_layers, nb_train_env=nb_eval_env,
                     nb_eval_env=nb_eval_env, hidden_size=hidden_size,
                     noise_config=eval_noise_config, eval_noise_config=eval_noise_config,
                     normalize_state=normalize_state, device=device)

    # load checkpoints if directory is not empty
    policy_file = join(logdir, 'best_policy.pt')
    R = - np.inf
    if exists(policy_file):
        state_dict = torch.load(policy_file)
        R = state_dict["return"]
        info(f"Loading policy with return {R}...")
        eval_policy.load_state_dict(state_dict)
    else:
        raise ValueError(f"{policy_file} does not exists, no policy available...")

    evaluate(env_config.dt, 0, eval_env, eval_policy, env_config.time_limit,
             eval_return=True, video=True, progress_bar=True)


if __name__ == '__main__':
    args = setup_args()

    policy_config, _, eval_noise_config, env_config = read_config(args)

    logto(args.logdir)
    main(
        logdir=args.logdir, hidden_size=args.hidden_size, nb_layers=args.nb_layers,
        nb_eval_env=args.nb_eval_env, policy_config=policy_config,
        eval_noise_config=eval_noise_config, env_config=env_config,
        normalize_state=args.normalize_state)
