"""Utilities."""
import torch
from torch import Tensor
from typing import Tuple
from memory.memorytrajectory import BatchTraj
import numpy as np
from abstract import ParametricFunction, Arrayable
from convert import check_array
from copy import deepcopy

def compute_return(rewards: Arrayable, dones: Arrayable):
    R = 0
    rewards = check_array(rewards)
    dones = check_array(dones)
    for r, d in zip(rewards[::-1], dones[::-1]):
        R = r + R * (1 - d)
    return np.mean(R)

def values(v_function: ParametricFunction, traj: BatchTraj,
           gamma: float, lambd: float, dt: float) -> Tuple[Tensor, Tensor]:
    """ Returns value and target value. """
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
