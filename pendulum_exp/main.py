"""Trying to solve pendulum using a robust advantage learning."""
from typing import Optional
import sys
from os.path import join, exists
from logging import info, basicConfig, INFO
import numpy as np
import torch
from tqdm import tqdm

from abstract import Policy, Env, Arrayable
from interact import interact
from evaluation import specific_evaluation
from utils import compute_return
from mylog import log, logto, log_video
from parse import setup_args
from config import configure

def train(nb_steps: int, env: Env, policy: Policy, start_obs: Arrayable):
    """ Trains for one epoch. """
    policy.train()

    obs = start_obs
    for _ in range(nb_steps):
        # interact
        obs, _, _ = interact(env, policy, obs)
    return obs

def evaluate(dt: float, epoch: int, env: Env, policy: Policy,
             time_limit: Optional[float] = None, eval_return: bool = False,
             progress_bar: bool = False, video: bool = False, no_log: bool = False):
    """ Evaluate. """
    log_gap = int(.1 / dt)
    policy.eval()

    R = None
    if eval_return:
        rewards, dones = [], []
        imgs = []
        time_limit = time_limit if time_limit else 10
        nb_steps = int(time_limit / dt)
        obs = env.reset()
        iter_range = tqdm(range(nb_steps)) if progress_bar else range(nb_steps)
        for _ in iter_range:
            obs, reward, done = interact(env, policy, obs)
            rewards.append(reward)
            dones.append(done)
            if video:
                imgs.append(env.render(mode='rgb_array'))
        R = compute_return(np.stack(rewards, axis=0),
                           np.stack(dones, axis=0))
        info(f"At epoch {epoch}, return: {R}")
        if not no_log:
            log("Return", R, epoch) # don't log when outputing video
        if video:
            log_video("demo", epoch, np.stack(imgs, axis=0))

    if not no_log:
        specific_evaluation(epoch, log_gap, dt, env, policy)
    return R

def main(args):
    """ Starts training. """
    logdir = args.logdir
    noreload = args.noreload
    dt = args.dt
    nb_epochs = args.nb_epochs
    nb_steps = args.nb_steps
    time_limit = args.time_limit

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy, env, eval_env = configure(args)
    policy = policy.to(device)

    obs = env.reset()

    # load checkpoints if directory is not empty
    policy_file = join(logdir, 'best_policy.pt')
    R = - np.inf
    cur_e = 0
    if exists(policy_file) and not noreload:
        state_dict = torch.load(policy_file)
        R = state_dict["return"]
        cur_e = state_dict["epoch"]
        info(f"Loading policy with return {R} at epoch {cur_e}...")
        policy.load_state_dict(state_dict)
    log_gap = int(.1 / dt)

    for e in range(cur_e, nb_epochs):
        info(f"Epoch {e}...")
        obs = train(nb_steps, env, policy, obs)
        new_R = evaluate(
            dt,
            e, eval_env,
            policy,
            time_limit,
            eval_return=e % log_gap == log_gap - 1,
        )
        if new_R is not None:
            # policy.observe_evaluation(new_R)
            if new_R > R:
                info(f"Saving new policy with return {new_R}")
                state_dict = policy.state_dict()
                state_dict["return"] = new_R
                state_dict["epoch"] = e
                torch.save(state_dict, policy_file)
                R = new_R
    env.close()
    eval_env.close()


if __name__ == '__main__':
    args = setup_args()

    # configure logging
    if args.redirect_stdout:
        basicConfig(filename=join(args.logdir, 'out.log'), level=INFO)
    else:
        basicConfig(stream=sys.stdout, level=INFO)

    logto(args.logdir, reload=not args.noreload)

    main(args)
