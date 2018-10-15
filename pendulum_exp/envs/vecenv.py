""" Vectorizing a list of environments (see openai baselines) """
from multiprocessing import Pipe, Process
from envs.utils import CloudpickleWrapper, VecEnv
from envs.utils import tile_images
import numpy as np

def worker(remote, env_wrapper):
    """
    :args remote: children side of pipe
    :args env_wrapper: pickled version of the environment
    """
    env = env_wrapper.x
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            o, r, d, i = env.step(data)
            if d:
                o = env.reset()
            remote.send((o, r, d, i))
        elif cmd == 'reset':
            o = env.reset()
            remote.send(o)
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'seed':
            remote.send(env.seed(data))
        else:
            raise NotImplementedError

class SubprocVecEnv(VecEnv):
    """
    Execute several environment parallely.

    :args envs: a list of SIMILAR environment to run parallely
    """
    def __init__(self, envs):
        self.waiting = False
        self.closed = False
        self.envs = envs
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env)))
                   for (work_remote, env) in zip(self.work_remotes, envs)]
        for p in self.ps:
            p.daemon = True # if main crashes, crash all
            p.start()
        for remote in self.work_remotes:
            remote.close() # work_remote are only used in child processes

        # get spaces
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(envs), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def seed(self, seeds):
        """ Seeding environment """
        for remote, s in zip(self.remotes, seeds):
            remote.send(('seed', s))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode='human'):
        self.remotes[0].send(('render', None))
        img = self.remotes[0].recv()
        if mode == 'rgb_array':
            return img
        elif mode != 'human':
            raise NotImplementedError

    def full_render(self, mode='human'):
        for remote in self.remotes:
            remote.send(('render', None))
        imgs = [remote.recv() for remote in self.remotes]
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

if __name__ == '__main__':
    from envs.pusher import PusherEnv
    nenvs = 64
    envs = [PusherEnv() for _ in range(nenvs)]
    vec_env = SubprocVecEnv(envs)

    obs = vec_env.reset()
    T = 200

    for i in range(T):
        a = [vec_env.action_space.sample() for _ in range(nenvs)]
        obs, rews, dones, _ = vec_env.step(a)
        vec_env.render()
