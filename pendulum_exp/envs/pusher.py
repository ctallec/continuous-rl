"""
Implement simple pusher environment.
"""
import gym
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
from abstract import Env

class AbstractPusher(gym.Env, Env):
    """ Abstract pusher class """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.observation_space = Box(low=-9999, high=9999, shape=(1,), dtype=np.float32)
        self._x = None
        self.dt = .1

        # rendering utilities
        self.seed()
        self.viewer = None
        self.pusher = None
        self.pushertrans = None

    @property
    def action_space(self):
        raise NotImplementedError()

    def action(self, action):
        raise NotImplementedError()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        action = self.action(action)
        self._x -= action * self.dt
        return self._x, np.exp(- self._x[0] ** 2), False, {}

    def reset(self):
        # without this, all thread have the same seed
        np.random.seed()
        self._x = np.random.normal(0, 1, (1,))

        return self._x

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = 6
        scale = screen_width / world_width
        pusherheight = 400
        pusherwidth = self.dt / world_width * scale

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -pusherwidth/2, pusherwidth/2, pusherheight, 0
            self.pusher = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.pushertrans = rendering.Transform()
            self.pusher.add_attr(self.pushertrans)
            self.viewer.add_geom(self.pusher)
            self.pusher.set_color(0., 1., 0.)

        if self._x is None:
            return None

        self.pushertrans.set_translation(self._x * scale + screen_width / 2, 0)
        g = (1 - min(max(self._x[0], -3), 3) ** 2 / 9)
        self.pusher.set_color(1-g, g, 0)
        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()

class DiscretePusherEnv(AbstractPusher): # pylint: disable=too-many-instance-attributes
    """Discrete pusher environment."""
    @property
    def action_space(self):
        return Discrete(3)

    def action(self, action):
        return action - 1

class ContinuousPusherEnv(AbstractPusher):
    """Continuous pusher environment."""
    @property
    def action_space(self):
        return Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def action(self, action):
        return action


if __name__ == '__main__':
    from pyglet.window import key

    a = [1]

    def key_press(k, _):
        """ What happens when a key is pressed """
        if k == key.LEFT:
            a[0] = 2
        if k == key.RIGHT:
            a[0] = 0

    def key_release(k, _):
        """ What happens when a key is released """
        if k == key.LEFT:
            a[0] = 1
        if k == key.RIGHT:
            a[0] = 1

    env = DiscretePusherEnv()
    env.render()

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a[0])
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(a[0]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            env.render()
            if done or restart:
                break
    env.close()
