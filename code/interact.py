"""Policy-env interactions."""
from abstract import Arrayable
from envs.env import Env
from policies.policy import Policy

def interact(
        env: Env,
        policy: Policy,
        start_obs: Arrayable):
    action = policy.step(start_obs)
    next_obs, reward, done, information = env.step(action)
    time_limit = information['time_limit'] if 'time_limit' in information else None
    policy.observe(next_obs, reward, done, time_limit)
    return next_obs, reward, done
