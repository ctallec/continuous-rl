"""Policy-env interactions."""
from typing import Tuple
from abstract import Arrayable
from envs.env import Env
from policies.policy import Policy
import numpy.ndarray as array

def interact(
        env: Env,
        policy: Policy,
        start_obs: Arrayable) -> Tuple[array, array, array]:
    """One step interaction between env and policy.

    :args env: environment
    :args policy: policy
    :args start_obs: initial observation

    :return: (next observation, reward, terminal?)
    """
    action = policy.step(start_obs)
    next_obs, reward, done, information = env.step(action)
    time_limit = information['time_limit'] if 'time_limit' in information else None
    policy.observe(next_obs, reward, done, time_limit)
    return next_obs, reward, done
