from typing import List, Optional
import random
import numpy as np
from numpy import array

from convert import check_array, check_tensor
from abstract import Arrayable, Tensorable
from cudaable import Cudaable

class Trajectory:
    def __init__(self,
                 obs: Optional[List[np.ndarray]] = None,
                 actions: Optional[List[np.ndarray]] = None,
                 rewards: Optional[List[float]] = None,
                 done: Optional[List[float]] = None,
                 time_limit: Optional[List[float]] = None,
                 boundlength: Optional[int] = None) -> None:
        """Stores a trajectory as a list of (obs, action, reward, done, time_limit).

        :args obs: initial list of obs
        :args actions: initial list of actions
        :args rewards: initial list of rewards
        :args done: initial list of done signal
        :args time_limit: initial list of time limits
        :args boundlength: max trajectory length
        """
        if obs is None:
            obs = []
        if actions is None:
            actions = []
        if rewards is None:
            rewards = []
        if done is None:
            done = []
        if time_limit is None:
            time_limit = []
        self._obs = obs
        self._actions = actions
        self._rewards = rewards
        self._done = done
        self._time_limit = time_limit
        self._boundlength = boundlength

        length = len(self)
        assert len(self._rewards) == length and len(self._actions) == length \
            and len(self._done) == length and len(self._time_limit) == length

    def push(self, obs: Arrayable, action: Arrayable, reward: float,
             done: float, time_limit: float) -> None:
        """
        Push a single transition on a trajectory
        (before seing the next observation).
        """
        obs = check_array(obs)
        action = check_array(action)

        self._obs.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)
        self._done.append(done)
        self._time_limit.append(time_limit)
        self.boundlength()

    def extract(self, length: int) -> "Trajectory":
        """Extract a random sub trajectory of length length."""
        assert length <= len(self)
        start = random.randrange(len(self) - length + 1)
        stop = start + length
        obs = self._obs[start:stop]
        actions = self._actions[start:stop]
        rewards = self._rewards[start:stop]
        done = self._done[start:stop]
        time_limit = self._time_limit[start:stop]
        return Trajectory(obs, actions, rewards, done, time_limit)

    def __len__(self) -> int:
        return len(self._obs)

    def boundlength(self) -> None:
        """Resize trajectory to boundlength if trajectory is too long."""
        if self._boundlength is None or len(self) <= self._boundlength:
            return None
        delta = max(0, len(self) - self._boundlength)
        self._obs = self._obs[delta:]
        self._actions = self._actions[delta:]
        self._rewards = self._rewards[delta:]
        self._done = self._done[delta:]
        self._time_limit = self._time_limit[delta:]
        assert len(self) == self._boundlength

    @property
    def isdone(self) -> bool:
        """True if current done is True."""
        if len(self) == 0:
            return False

        return self._done[-1] == 1.

    @staticmethod
    def tobatch(*trajs: "Trajectory") -> "BatchTraj":
        """Turn a list of trajs into a batch of trajs."""
        batch_size = len(trajs)
        length_traj = len(trajs[0])
        assert all(len(traj) == length_traj for traj in trajs)

        obs_shape = trajs[0]._obs[0].shape
        action_shape = trajs[0]._actions[0].shape

        obs = np.zeros((batch_size, length_traj, *obs_shape))
        actions = np.zeros((batch_size, length_traj, *action_shape))
        rewards = np.zeros((batch_size, length_traj))
        done = np.zeros((batch_size, length_traj))
        time_limit = np.zeros((batch_size, length_traj))
        for i, traj in enumerate(trajs):
            assert obs[i, :].shape == array(traj._obs).shape \
                and actions[i, :].shape == array(traj._actions).shape
            obs[i, :] = array(traj._obs)
            actions[i, :] = array(traj._actions)
            rewards[i, :] = array(traj._rewards)
            done[i, :] = array(traj._done)

        return BatchTraj(obs=obs, actions=actions, rewards=rewards,
                         done=done, time_limit=time_limit)


class BatchTraj(Cudaable):
    """Batched trajectory."""
    def __init__(self, obs: Tensorable, actions: Tensorable,
                 rewards: Tensorable, done: Tensorable,
                 time_limit: Tensorable) -> None:
        self.obs = check_tensor(obs)
        self.actions = check_tensor(actions)
        self.rewards = check_tensor(rewards)
        self.done = check_tensor(done)
        self.time_limit = check_tensor(time_limit)
        self.batch_size = self.obs.shape[0]
        self.length = self.obs.shape[1]

        assert self.actions.shape[0] == self.batch_size \
            and self.rewards.shape[0] == self.batch_size \
            and self.done.shape[0] == self.batch_size \
            and self.time_limit.shape[0] == self.batch_size
        assert self.actions.shape[1] == self.length \
            and self.rewards.shape[1] == self.length \
            and self.done.shape[1] == self.length \
            and self.time_limit.shape[1] == self.length
        assert len(self.done.shape) == 2 \
            and len(self.rewards.shape) == 2 \
            and len(self.time_limit.shape) == 2

    def to(self, device) -> "BatchTraj":
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.done = self.done.to(device)
        self.time_limit = self.time_limit.to(device)
        return self

    def __getitem__(self, key) -> "BatchTraj":
        return BatchTraj(obs=self.obs[key], actions=self.actions[key],
                         rewards=self.rewards[key], done=self.done[key],
                         time_limit=self.time_limit[key])

    @property
    def device(self):
        return self.obs.device

class MemoryTrajectory:
    def __init__(self, maxsize: int, memory: Optional[List[Trajectory]] = None) -> None:
        self._maxsize = maxsize
        if memory is None:
            memory = []
        self._memory: List[Trajectory] = memory
        self._cumsizes: List[int] = np.cumsum([len(traj) for traj in self._memory]).tolist()

    def __len__(self) -> int:
        return len(self._memory)

    @property
    def size(self) -> int:
        if len(self._cumsizes) == 0:
            return 0
        return self._cumsizes[-1]

    def _reducesize(self) -> None:
        if self.size < self._maxsize:
            return None
        i, subsize = next((i, cumsize_i) for i, cumsize_i in enumerate(self._cumsizes)
                          if cumsize_i > self.size - self._maxsize)
        self._memory = self._memory[i+1:]
        self._cumsizes = [s - subsize for s in self._cumsizes[i+1:]]

    def push(self, trajectory: Trajectory) -> None:
        self._memory.append(trajectory)
        self._cumsizes.append(self.size + len(trajectory))
        self._reducesize()

    def choose(self, idxs: List[int]) -> List[Trajectory]:
        return [self._memory[i] for i in idxs]


class MemorySampler:
    def __init__(self, memory: MemoryTrajectory,
                 batch_size: int, length_traj: int) -> None:
        self._memory = memory
        self._batch_size = batch_size
        self._length_traj = length_traj

    def sample_batch(self) -> BatchTraj:
        idxs = random.sample([i for i in range(len(self._memory))], self._batch_size)
        trajs = self._memory.choose(idxs)
        trajs_truncated = [traj.extract(self._length_traj) for traj in trajs]

        obs_shape = trajs_truncated[0]._obs[0].shape
        action_shape = trajs_truncated[0]._actions[0].shape

        obs = np.zeros((self._batch_size, self._length_traj, *obs_shape))
        actions = np.zeros((self._batch_size, self._length_traj, *action_shape))
        rewards = np.zeros((self._batch_size, self._length_traj))
        done = np.zeros((self._batch_size, self._length_traj))
        tl = np.zeros((self._batch_size, self._length_traj))

        for i, traj in enumerate(trajs_truncated):
            for t in range(self._length_traj):
                obs[i, t] = traj._obs[t]
                actions[i, t] = traj._actions[t]
                rewards[i, t] = traj._rewards[t]
                done[i, t] = traj._done[t]
                tl[i, t] = traj._time_limit[t]

        batchtraj = BatchTraj(obs=obs, actions=actions, rewards=rewards,
                              done=done, time_limit=tl)
        return batchtraj

    def warmed_up(self) -> bool:
        return len(self._memory) > self._batch_size
