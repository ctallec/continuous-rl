from typing import Optional, List
from abstract import Arrayable
from stateful import StateDict
from policies.policy import Policy
from stateful import CompoundStateful
import numpy as np
from torch import Tensor
from convert import th_to_arr, check_array
from memory.memorytrajectory import Trajectory
from actors.online_actor import OnlineActor, OnlineActorContinuous
from critics.online_critic import OnlineCritic
from abc import abstractmethod


class OnlinePolicy(CompoundStateful, Policy):
    def __init__(
            self, T: int, nb_train_env: int,
            actor: OnlineActor, critic: OnlineCritic) -> None:
        CompoundStateful.__init__(self)

        # reset and set in train mode
        self.train()

        # define learning components
        self._actor = actor
        self._critic = critic
        self._nb_train_env = nb_train_env
        self._count = 0
        self._T = T
        self._device="cpu"
        self.reset()

    def step(self, obs: Arrayable) -> np.ndarray:
        if self._train:
            action = th_to_arr(self._actor.act_noisy(obs))
        else:
            action = th_to_arr(self._actor.act(obs))

        self._current_obs = check_array(obs)
        self._current_action = check_array(action)
        # TODO check continuous...

        if isinstance(self._actor, OnlineActorContinuous):
            action = np.clip(action, -1, 1)
        return action

    def reset(self) -> None:
        self._current_trajectories: List[Trajectory] = \
            [Trajectory(boundlength=self._T) for _ in range(self._nb_train_env)]
        self._current_obs = np.array([])
        self._current_action = np.array([])

    def observe(self,
                next_obs: Arrayable,
                reward: Arrayable,
                done: Arrayable,
                time_limit: Optional[Arrayable] = None) -> None:

        if not self._train:
            return None

        self._count += 1
        reward = check_array(reward)
        done = check_array(done)
        if time_limit is None:
            time_limit = np.zeros(done.shape)
        time_limit = check_array(time_limit)

        for k, traj in enumerate(self._current_trajectories):
            traj.push(self._current_obs[k], self._current_action[k], reward[k],
                      float(done[k]), float(time_limit[k]))

        self.learn()

    @abstractmethod
    def learn(self) -> None:
        pass

    def state_dict(self) -> StateDict:
        state = CompoundStateful.state_dict(self)
        state["count"] = self._count
        return state

    def load_state_dict(self, state_dict: StateDict):
        CompoundStateful.load_state_dict(self, state_dict)
        self._count = state_dict["count"]

    def train(self) -> None:
        self._train = True

    def eval(self) -> None:
        self._train = False

    def value(self, obs: Arrayable) -> Tensor:
        return self._critic.value(obs)

    def actions(self, obs: Arrayable) -> Tensor:
        return self._actor.actions(obs)

    def to(self, device) -> "OnlinePolicy":
        self._device = device
        CompoundStateful.to(self, device)
        return self
        
