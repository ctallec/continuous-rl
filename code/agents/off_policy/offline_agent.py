from typing import Optional
from abstract import Arrayable, Tensorable
from cudaable import Cudaable
from stateful import StateDict
from memory.utils import setup_memory
from stateful import CompoundStateful
from mylog import log
from logging import info
import numpy as np
from torch import Tensor
from convert import th_to_arr, arr_to_th
from agents.agent import Agent
from critics.critic import Critic
from actors.actor import Actor

class OfflineAgent(CompoundStateful, Agent, Cudaable):
    """Offline agent, i.e. learning from a buffer.

    :args memory_size: size of the memory
    :args batch_size: size of training batches
    :args steps_btw_train: number of interactions with the environment
        between two training steps
    :args learn_per_step: number of batches processed during a training
       step
    :args alpha:
    :args beta: prioritized experience replay parameters (untested)
    :args actor: actor used
    :args critic: critic used
    """
    def __init__(
            self, memory_size: int, batch_size: int,
            steps_btw_train: int, learn_per_step: int,
            alpha: Optional[float], beta: Optional[float],
            actor: Actor, critic: Critic) -> None:
        CompoundStateful.__init__(self)

        # reset and set in train mode
        self.reset()
        self.train()

        # define learning components
        self._actor = actor
        self._critic = critic
        self._sampler = setup_memory(
            alpha=alpha, beta=beta, memory_size=memory_size, batch_size=batch_size)

        # counter and parameters
        self._count = 0
        self._warm_up = 10 # prevents learning from a near empty buffer
        self._steps_btw_train = steps_btw_train
        self._learn_per_step = learn_per_step

    def reset(self):
        """Reset current transition (not memory buffer) !"""
        # internals
        self._obs = np.array([])
        self._action = np.array([])
        self._reward = np.array([])
        self._next_obs = np.array([])
        self._done = np.array([])
        self._time_limit = np.array([])

    def step(self, obs: Arrayable):
        if self._train:
            self._obs = obs
            action = self._actor.act_noisy(obs)
            self._action = action
        else:
            action = th_to_arr(self._actor.act(obs))

        return action

    def observe(self,
                next_obs: Arrayable,
                reward: Arrayable,
                done: Arrayable,
                time_limit: Optional[Arrayable] = None):
        """If in train mode, store transition in buffer, and may perform a training step."""
        if self._train:
            self._count += 1
            self._next_obs = next_obs
            self._reward = reward
            self._done = done
            self._time_limit = time_limit
            self._sampler.push(
                self._obs, self._action, self._next_obs,
                self._reward, self._done, self._time_limit)
            self.learn()

    def learn(self):
        if self._count % self._steps_btw_train == self._steps_btw_train - 1 and self._count > self._warm_up:
            cum_critic_loss = 0
            cum_critic_value = 0

            for _ in range(self._learn_per_step):
                obs, action, next_obs, reward, done, weights, time_limit = \
                    self._sampler.sample()

                # don't update when a time limit is reached
                if time_limit is not None:
                    weights = weights * (1 - time_limit)

                max_action = self._actor.act(obs)
                max_next_action = self._actor.act(next_obs, target=True)

                critic_loss = self._critic.optimize(
                    obs, action, max_action,
                    next_obs, max_next_action, reward, done, time_limit, weights)
                critic_value = self._critic.critic(
                    obs, max_action)

                weights = arr_to_th(weights, device=critic_loss.device)
                self._actor.optimize(-critic_value)
                self._sampler.observe(th_to_arr(critic_loss * weights))

                cum_critic_loss += (critic_loss * weights).mean().item()
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

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def to(self, device):
        return CompoundStateful.to(self, device)

    def value(self, obs: Arrayable) -> Tensor:
        return self._critic.value(obs, self._actor)

    def actions(self, obs: Arrayable) -> Tensor:
        return self._actor.act(obs)

    def advantage(self, obs: Arrayable, action: Tensorable) -> Tensor:
        return self._critic.advantage(obs, action, self._actor)
