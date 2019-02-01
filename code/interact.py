"""Policy-env interactions."""
from abstract import Env, Arrayable
from policies.policy import Policy

def interact(
        env: Env,
        policy: Policy,
        start_obs: Arrayable):
    action = policy.step(start_obs)
    next_obs, reward, done, info = env.step(action)
    time_limit = info['time_limit'] if 'time_limit' in info else None
    policy.observe(next_obs, reward, done, time_limit)
    return next_obs, reward, done
