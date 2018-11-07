"""Makes bipedal walker hardcore easier."""
from gym.envs.box2d import bipedal_walker

class WalkerHardcore(bipedal_walker.BipedalWalkerHardcore):
    def __init__(self, dt: float) -> None:
        bipedal_walker.FPS = 1. / dt
        super().__init__()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if reward == -100:
            reward = 0
        return obs, reward, done, info
