"""Perform standalone evaluation."""
from os.path import join, exists
from logging import info
import torch
import numpy as np

from main import evaluate
from parse import setup_args
from config import configure
from mylog import logto


def main(args):
    """Evaluation corresponding to given argparse arguments."""
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent, env, eval_env = configure(args)
    agent = agent.to(device)

    # load checkpoints if directory is not empty
    agent_file = join(args.logdir, 'best_agent.pt')
    R = - np.inf
    if exists(agent_file):
        state_dict = torch.load(agent_file)
        R = state_dict["return"]
        info(f"eval> Loading agent with return {R}...")
        agent.load_state_dict(state_dict)
    else:
        raise ValueError(f"{agent_file} does not exists, no agent available...")

    evaluate(args.dt, 0, eval_env, agent, args.time_limit,
             eval_return=True, video=True, progress_bar=True, no_log=True)


if __name__ == '__main__':
    args = setup_args()

    try:
        logto(args.logdir)
    except Exception:
        pass # Errors can occur with pickle. Still try to output video even if pickle is corrupted
    main(args)
