"""Main training file"""
from typing import Optional
import sys
from os.path import join, exists
from logging import info, basicConfig, INFO
import numpy as np
import torch
from tqdm import tqdm

from abstract import Arrayable
from envs.env import Env
from agents.agent import Agent
from agents.on_policy.online_agent import OnlineAgent
from interact import interact
from evaluation import specific_evaluation
from utils import compute_return
from mylog import log, logto, log_video
from parse import setup_args
from config import configure

def train(nb_steps: int, env: Env, agent: Agent, start_obs: Arrayable):
    """Trains for one epoch.

    :args nb_steps: number of interaction steps
    :args env: environment
    :args agent: interacting agent
    :start_obs: starting observation

    :return: final observation
    """
    agent.train()
    agent.reset()
    obs = start_obs
    for _ in range(nb_steps):
        # interact
        obs, _, _ = interact(env, agent, obs)
    return obs

def evaluate(dt: float, epoch: int, env: Env, agent: Agent, eval_gap: float, # noqa: C901
             time_limit: Optional[float] = None, eval_return: bool = False,
             progress_bar: bool = False, video: bool = False, no_log: bool = False,
             test: bool = False, eval_policy: bool = True) -> Optional[float]:
    """Evaluate agent in environment.

    :args dt: time discretization
    :args epoch: index of the current epoch
    :args env: environment
    :args agent: interacting agent
    :args eval_gap: number of normalized epochs (epochs divided by dt)
        between training steps
    :args time_limit: maximal physical time (number of steps divided by dt)
        spent in the environment
    :args eval_return: do we only perform specific evaluation?
    :args progress_bar: use a progress bar?
    :args video: log a video of the interaction?
    :args no_log: do we log results
    :args test: log to a different test summary
    :args eval_policy: if the exploitation policy is noisy,
        remove the noise before evaluating

    :return: return evaluated, None if no return is evaluated
    """
    log_gap = int(eval_gap / dt)
    agent.eval()
    if not eval_policy and isinstance(agent, OnlineAgent):
        agent.noisy_eval()
    agent.reset()
    R = None
    if eval_return:
        rewards, dones = [], []
        imgs = []
        time_limit = time_limit if time_limit else 10
        nb_steps = int(time_limit / dt)
        info(f"eval> evaluating on a physical time {time_limit}"
             f" ({nb_steps} steps in total)")
        obs = env.reset()
        iter_range = tqdm(range(nb_steps)) if progress_bar else range(nb_steps)
        for _ in iter_range:
            obs, reward, done = interact(env, agent, obs)
            rewards.append(reward)
            dones.append(done)
            if video:
                imgs.append(env.render(mode='rgb_array'))
        R = compute_return(np.stack(rewards, axis=0),
                           np.stack(dones, axis=0))
        tag = "noisy" if not eval_policy else ""
        info(f"eval> At epoch {epoch}, {tag} return: {R}")
        if not no_log:
            if not eval_policy:
                log("Return_noisy", R, epoch)
            elif not video: # don't log when outputing video
                if not test:
                    log("Return", R, epoch)
                else:
                    log("Return_test", R, epoch)
        if video:
            log_video("demo", epoch, np.stack(imgs, axis=0))

    if not no_log:
        specific_evaluation(epoch, log_gap, dt, env, agent)
    return R

def main(args):
    """Main training procedure."""
    logdir = args.logdir
    noreload = args.noreload
    dt = args.dt
    nb_true_epochs = args.nb_true_epochs
    nb_steps = args.nb_steps
    time_limit = args.time_limit
    eval_gap = args.eval_gap

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent, env, eval_env = configure(args)
    agent = agent.to(device)

    obs = env.reset()

    # load checkpoints if directory is not empty
    agent_file = join(logdir, 'best_agent.pt')
    R = - np.inf
    cur_e = 0
    if exists(agent_file) and not noreload:
        state_dict = torch.load(agent_file)
        R = state_dict["return"]
        cur_e = state_dict["epoch"]
        info(f"train> Loading agent with return {R} at epoch {cur_e}...")
        agent.load_state_dict(state_dict)
    log_gap = max(int(eval_gap / dt), 1)
    info(f"train> number of epochs between evaluations: {log_gap}")
    snapshot_gap = 10 * max(int(eval_gap / dt), 1)
    info(f"train> number of epochs between snapshots: {snapshot_gap}")

    for e in range(cur_e, int(nb_true_epochs / dt)):
        info(f"train> Epoch {e}...")
        obs = train(nb_steps, env, agent, obs)
        new_R = evaluate(
            dt,
            e, eval_env,
            agent,
            eval_gap,
            time_limit,
            eval_return=e % log_gap == log_gap - 1,
            test=False
        )

        # evaluate with noisy actions
        evaluate(
            dt,
            e, eval_env,
            agent,
            eval_gap,
            time_limit,
            eval_return=e % log_gap == log_gap - 1,
            test=False,
            eval_policy=False
        )

        if args.snapshot and e % snapshot_gap == snapshot_gap - 1:
            state_dict = agent.state_dict()
            state_dict["return"] = new_R
            state_dict["epoch"] = e
            snap_file = join(logdir, f'agent_{e}.pt')
            torch.save(state_dict, snap_file)
        if new_R is not None:
            # agent.observe_evaluation(new_R)
            if new_R > R:
                evaluate(
                    dt, e, eval_env, agent, eval_gap,
                    time_limit, eval_return=True, test=True)
                info(f"train> Saving new agent with return {new_R}")
                state_dict = agent.state_dict()
                state_dict["return"] = new_R
                state_dict["epoch"] = e
                torch.save(state_dict, agent_file)
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
