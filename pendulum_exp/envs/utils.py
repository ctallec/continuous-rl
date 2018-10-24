""" Environment utilities """
from abc import ABC, abstractmethod
import gym
from gym import ActionWrapper
from gym.spaces import Discrete, Box
import numpy as np
from envs.pusher import PusherEnv

def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

class AlreadySteppingError(Exception):
    """
    Raised when an asynchronous step is running while
    step_async() is called again.
    """
    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)

class NotSteppingError(Exception):
    """
    Raised when an asynchronous step is not running but
    step_wait() is called.
    """
    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
        - obs: an array of observations, or a tuple of
        arrays of observations.
        - rews: an array of rewards
        - dones: an array of "episode done" booleans
        - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        """ One environment step

        :args actions: np array of actions

        :returns: obses, rews, dones, infos
        """
        self.step_async(actions)
        return self.step_wait()

    @abstractmethod
    def render(self, mode='human'):
        """ Renders """
        pass

    @property
    def unwrapped(self):
        """ Unwraps the environment """
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped # pylint: disable=E1101
        return self

class VecEnvWrapper(VecEnv):
    """ Wraps a vectorized environment """
    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        VecEnv.__init__(self,
                        num_envs=venv.num_envs,
                        observation_space=observation_space or venv.observation_space,
                        action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        self.venv.render(mode)

class CloudpickleWrapper: # pylint: disable=R0903
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class WrapPendulum(ActionWrapper):
    """ Wrap pendulum. """
    def action_space(self):
        return Discrete(2)

    def action(self, action):
        return 4 * np.array(action)[np.newaxis] - 2

class WrapContinuousPendulum(ActionWrapper):
    """ Wrap Continuous Pendulum. """
    def action_space(self):
        return Box(low=-1, high=1, shape=(1,))

    def action(self, action):
        return 2 * action

def make_env(env_id, dt):
    """ Make env. """
    # pendulum
    if env_id == 'pendulum':
        env = gym.make('Pendulum-v0').unwrapped
        env.dt = dt
        return WrapPendulum(env)
    if env_id == 'cartpole':
        env = gym.make('CartPole-v1').unwrapped
        env.tau = dt
        return env
    if env_id == 'continuous_pendulum':
        env = gym.make('Pendulum-v0').unwrapped
        env.dt = dt
        return WrapContinuousPendulum(env)
    env = PusherEnv()
    env.dt = dt
    return env
