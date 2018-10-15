"""Policy-env interactions."""
from abstract import Env, Policy, Arrayable


def interact(
        env: Env,
        policy: Policy,
        start_obs: Arrayable):
    action = policy.step(start_obs)
    next_obs, reward, done, _ = env.step(action)
    policy.observe(next_obs, reward, done)
    return next_obs, reward, done
