from typing import List, Optional
from abstract import Arrayable
import random


class Trajectory:
    def __init__(self, 
                 obs: Optional[List[Arrayable]] = None, 
                 actions: Optional[List[Arrayable]] = None, 
                 rewards: Optional[List[float]] = None,
                 done: Optional[List[bool]] = None) -> None:
        
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

    def push(self, obs: Arrayable, action: Arrayable, reward: float, done: bool) -> None:
        assert not self.isdone

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

        return self._done[-1]
    

class MemoryTrajectory:
    def __init__(self, size: int, batch_size: int, memory: Optional[List[Trajectory]] = None) -> None:
        self._size = size
        self._batch_size = batch_size

        if memory is None:
            memory = []
        self._memory: List[Trajectory] = memory

        self._idx = 0

    def __len__(self) -> int:
        return len(self._memory)

    def size(self) -> int:
        return sum(len(traj) for traj in self._memory)


    def push(self, trajectory: Trajectory):
        pass







