from typing import List, Optional, Tuple, NamedTuple
from collections import namedtuple
import random
import numpy as np
from torch import Tensor


from convert import check_array, check_tensor
from abstract import Arrayable, Tensorable
from cudaable import Cudaable




class Trajectory:
    def __init__(self, 
                 obs: Optional[List[np.ndarray]] = None, 
                 actions: Optional[List[np.ndarray]] = None, 
                 rewards: Optional[List[float]] = None,
                 done: Optional[List[float]] = None) -> None:
        
        if obs is None:
            obs = []
        if actions is None:
            actions = []
        if rewards is None:
            rewards = []
        if done is None:
            done = []
        self._obs = obs
        self._actions = actions
        self._rewards = rewards
        self._done = done

        l = len(self)
        assert len(self._rewards) == l and len(self._actions) == l and len(self._done) == l

    def push(self, obs: Arrayable, action: Arrayable, reward: float, done: float) -> None:
        assert not self.isdone
        obs = check_array(obs)
        action = check_array(action)

        self._obs.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)
        self._done.append(done)

    def extract(self, length: int) -> "Trajectory":
        assert length <= len(self)
        start = random.randrange(len(self) - length + 1)
        stop = length - start
        obs = self._obs[start:stop]
        actions = self._actions[start:stop]
        rewards = self._rewards[start:stop]
        done = self._done[start:stop]

        return Trajectory(obs, actions, rewards, done)


    def __len__(self) -> int:
        return len(self._obs)

    @property
    def isdone(self) -> bool:
        if len(self) == 0:
            return False

        return self._done[-1] == 1.
    

class BatchTraj(Cudaable):
    def __init__(self, obs: Tensorable, actions: Tensorable, 
                 rewards: Tensorable, done: Tensorable):
        self.obs = check_tensor(obs)
        self.actions = check_tensor(actions)
        self.rewards = check_tensor(rewards)
        self.done = check_tensor(done)
        self.batch_size = self.obs.shape[0]
        self.length = self.obs.shape[1] 

        assert self.actions.shape[0] == self.batch_size and self.rewards.shape[0] == self.batch_size \
            and self.done.shape[0] == self.batch_size
        assert self.actions.shape[1] == self.length and self.rewards.shape[1] == self.length \
            and self.done.shape[1] == self.length
    
    def to(self, device) -> "BatchTraj":
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.actions.to(device)
        self.done = self.done.to(device)
        return self

    def splitlast(self) -> Tuple["BatchTraj", Tuple[Tensor, Tensor, Tensor, Tensor]]:
        trunctraj = BatchTraj(self.obs[:,:-1], self.actions[:,:-1], self.rewards[:,:-1], self.done[:,:-1])
        last = (self.obs[:,-1], self.actions[:,-1], self.rewards[:,-1], self.done[:,-1])
        return trunctraj, last



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
        return self._cumsizes[-1]


    def _reducesize(self) -> None:
        if self.size < self._maxsize:
            return None
        i, subsize = next((i, cumsize_i) for i, cumsize_i in enumerate(self._cumsizes) \
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
    def __init__(self, memory: MemoryTrajectory, batch_size: int, length_traj: int) -> None:
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

        for i, traj in enumerate(trajs_truncated):
            for t in range(self._length_traj):
                obs[i,t] = traj._obs[t]
                actions[i,t] = traj._actions[t]
                rewards[i,t] = traj._rewards[t]
                done[i,t] = traj._done[t]

        batchtraj = BatchTraj(obs=obs, actions=actions, rewards=rewards, done=done)
        return batchtraj

    def warmed_up(self) -> bool:
        return len(self._memory) > self._batch_size








