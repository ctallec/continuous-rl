from abc import abstractmethod
from typing import Optional, List

import numpy as np
from torch import Tensor

from abstract import Arrayable
from stateful import StateDict
from stateful import CompoundStateful
from memory.trajectory import Trajectory
from convert import th_to_arr, check_array

from agents.agent import Agent
from actors.on_policy.online_actor import OnlineActor, OnlineActorContinuous
from critics.on_policy.online_critic import OnlineCritic


class OnlineAgent(CompoundStateful, Agent):
    """Abstraction for Online Agent.

    :args T: number of max steps used for bootstrapping
       (to be computationnally efficient, bootstrapping horizon is variable).
    :args actor: actor used
    :args critic: critic used
    """
    def __init__(
            self, T: int, actor: OnlineActor, critic: OnlineCritic) -> None:
        CompoundStateful.__init__(self)

        # reset and set in train mode
        self.train()

        # define learning components
        self._actor = actor
        self._critic = critic
        self._count = 0
        self._T = T
        self._device = "cpu"
        self.reset()

        # init _nb_train_env and _current_trajectories to None
        self._nb_train_env: Optional[int] = None
        self._current_trajectories: List[Trajectory] = []

    def step(self, obs: Arrayable) -> np.ndarray:
        if self._mode != "eval":
            action = th_to_arr(self._actor.act_noisy(obs))
        else:
            action = th_to_arr(self._actor.act(obs))

        self._current_obs = check_array(obs)
        self._current_action = check_array(action)
        if isinstance(self._actor, OnlineActorContinuous):
            action = np.clip(action, -1, 1)
        return action

    def reset(self) -> None:
        # use _nb_train_env to know if current trajectories were initialized at
        # some point
        if self._nb_train_env is not None:
            self._current_trajectories: List[Trajectory] = \
                [Trajectory(boundlength=self._T) for _ in range(self._nb_train_env)]
        self._current_obs = np.array([])
        self._current_action = np.array([])

    def observe(self,
                next_obs: Arrayable,
                reward: Arrayable,
                done: Arrayable,
                time_limit: Optional[Arrayable] = None) -> None:

        if self._mode != "train":
            return None

        self._count += 1
        reward = check_array(reward)
        done = check_array(done)
        if time_limit is None:
            time_limit = np.zeros(done.shape)
        time_limit = check_array(time_limit)

        if not self._current_trajectories:
            self._nb_train_env = done.shape[0]
            self._current_trajectories = \
                [Trajectory(boundlength=self._T) for _ in range(self._nb_train_env)]

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
        self._mode = "train"

    def eval(self) -> None:
        self._mode = "eval"

    def noisy_eval(self) -> None:
        self._mode = "noisy_eval"

    def value(self, obs: Arrayable) -> Tensor:
        return self._critic.value(obs)

    def actions(self, obs: Arrayable) -> Tensor:
        return self._actor.actions(obs)

    def to(self, device) -> "OnlineAgent":
        self._device = device
        CompoundStateful.to(self, device)
        return self
