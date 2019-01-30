"""Utilities."""
import torch
import numpy as np
from abstract import ParametricFunction, Arrayable
from convert import check_array

def compute_return(rewards: Arrayable, dones: Arrayable):
    R = 0
    rewards = check_array(rewards)
    dones = check_array(dones)
    for r, d in zip(rewards[::-1], dones[::-1]):
        R = r + R * (1 - d)
    return np.mean(R)
