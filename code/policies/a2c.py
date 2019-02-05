from typing import Optional, List
from abstract import Arrayable, Tensorable
from cudaable import Cudaable
from stateful import StateDict
from policies.policy import Policy
from stateful import CompoundStateful
from mylog import log
from logging import info
import numpy as np
from torch import Tensor
from convert import th_to_arr, check_array
from memory.memorytrajectory import MemoryTrajectory, MemorySampler, Trajectory
from actors.a2cactor import A2CActor
from critics.a2ccritic import A2CCritic


class A2CPolicy(CompoundStateful, Policy):
    def __init__(
            self, memory_size: int, batch_size: int, n_step: int,
            steps_btw_train: int, learn_per_step: int, nb_train_env: int,
            actor: A2CActor, critic: A2CCritic) -> None:
        CompoundStateful.__init__(self)

        # reset and set in train mode
        self.train()

        # define learning components
        self._actor = actor
        self._critic = critic
        self._memory = MemoryTrajectory(memory_size)
        self._sampler = MemorySampler(self._memory, batch_size, n_step)

        # # counter and parameters
        self._count = 0
        self._warm_up = 10 # prevents learning from a near empty buffer
        self._steps_btw_train = steps_btw_train
        self._learn_per_step = learn_per_step
        self._nb_train_env = nb_train_env

        self.reset()
 

    def step(self, obs: Arrayable) -> np.ndarray:
        if self._train:
            action = th_to_arr(self._actor.act_noisy(obs))
        else:
            action = th_to_arr(self._actor.act(obs))

        action = np.clip(action, -1, 1)
        self._current_obs = check_array(obs)
        self._current_action = check_array(action)
        return action

    def reset(self) -> None:
        # internals
        self._current_trajectories: List[Trajectory] = [Trajectory() for _ in range(self._nb_train_env)]
        self._current_obs = np.array([])
        self._current_action = np.array([])
        # self._time_limit = np.array([])

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
        time_limit = check_array(time_limit)
        for k in range(self._nb_train_env):
            traj = self._current_trajectories[k]
            traj.push(self._current_obs[k], self._current_action[k], reward[k], float(done[k]))
            if traj.isdone or (time_limit is not None and time_limit[k]):
                self._memory.push(traj)
                self._current_trajectories[k] = Trajectory()
        self._current_obs = check_array(next_obs)

        self.learn()

    def learn(self) -> None:
        if (self._count + 1) % self._steps_btw_train != 0 or not self._sampler.warmed_up():
            if (self._count + 1) % self._steps_btw_train == 0:
                info(f"NotWarmedup")
            return None

        info("Learn")
        cum_critic_loss = 0
        cum_critic_value = 0

        for _ in range(self._learn_per_step):
            traj = self._sampler.sample_batch()
            # don't update when a time limit is reached
            # if time_limit is not None:
            #     weights = weights * (1 - time_limit)

            critic_loss = self._critic.optimize(traj)
            critic_value = self._critic.critic(traj, target=True)
            
            # weights = arr_to_th(weights, device=critic_loss.device)
            obs, action = traj.obs[:,0], traj.actions[:,0]
            self._actor.optimize(obs, action, -critic_value)
            # self._sampler.observe(th_to_arr(critic_loss * weights))

            cum_critic_loss += critic_loss.mean().item()
            cum_critic_value += critic_value.mean().item()

        info(f'At step {self._count}, critic loss: {cum_critic_loss / self._learn_per_step}')
        info(f'At step {self._count}, critic value: {cum_critic_value / self._learn_per_step}')
        log("loss/critic", cum_critic_loss / self._learn_per_step, self._count)
        log("value/critic", cum_critic_value / self._learn_per_step, self._count)
        self._actor.log()
        self._critic.log()

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

    # def to(self, device):
    #     return CompoundStateful.to(self, device)

    # def value(self, obs: Arrayable) -> Tensor:
    #     return self._critic.value(obs, self._actor)

    # def actions(self, obs: Arrayable) -> Tensor:
    #     return self._actor.act(obs)

    

