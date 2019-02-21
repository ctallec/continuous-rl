"""Utilities."""
import torch
from torch import Tensor
from typing import Tuple
from memory.trajectory import BatchTraj
import numpy as np
from abstract import ParametricFunction, Arrayable
from convert import check_array
from copy import deepcopy

def compute_return(rewards: Arrayable, dones: Arrayable) -> float:
    """Compute return from rewards and termination signals.

    :args rewards: (seq_len, batch_size) reward array
    :args dones: (seq_len, batch_size) termination signal array

    :return: averaged undiscounted return
    """
    R = 0
    rewards = check_array(rewards)
    dones = check_array(dones)
    for r, d in zip(rewards[::-1], dones[::-1]):
        R = r + R * (1 - d)
    return np.mean(R)

def values(v_function: ParametricFunction, traj: BatchTraj,
           gamma: float, lambd: float, dt: float) -> Tuple[Tensor, Tensor]:
    """ Returns values and target values computed using GAE.

    Computes both the current value estimate on each state of the traj
    using the v_function, and a bootstrapped estimate of the value using
    generalized advantage estimation.

    :args v_function: Parametric V function
    :args traj: batch of trajectories on which to compute the values and target values
    :args gamma: discount factor
    :args lambd: truncated eligibility traces parameter

    :return: (values (batch_size, seq_len), target_values (batch_size, seq_len))
    """
    v = v_function(traj.obs).squeeze(-1)
    r = traj.rewards * dt
    d = traj.done
    tl = deepcopy(traj.time_limit)
    tl[..., -1] = 1
    stop = torch.max(d, tl)

    gae = (1 - tl) * \
        (r + gamma * (1 - d) * torch.cat([v[..., 1:], v[..., :1]], dim=-1) - v)

    for t in reversed(range(0, traj.length - 1)):
        gae[..., t] += (1 - stop[..., t]) * (gamma * lambd * gae[..., t + 1])

    return v, (v + gae).detach()
