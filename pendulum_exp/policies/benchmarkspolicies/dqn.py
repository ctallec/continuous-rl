from itertools import chain
import torch
import numpy as np
from abstract import ParametricFunction, Arrayable, Noise, StateDict, Policy

import copy
from convert import arr_to_th, check_array, th_to_arr
# from stats import penalize_mean
from config import DQNConfig
# from policies.shared import SharedAdvantagePolicy
from mylog import log
from logging import info
from optimizer import setup_optimizer
from memory.utils import setup_memory

class DQNPolicy(Policy):
    def __init__(self,
                 qnet_function: ParametricFunction,
                 qnet_noise: Noise,
                 policy_config: DQNConfig,
                 device) -> None:
        # super().__init__(policy_config)
        self._qnet_function = qnet_function.to(device)

        self._target_function = copy.deepcopy(self._qnet_function)

        self._qnet_noise = qnet_noise

        # config
        self._gamma = policy_config.gamma
        self._dt = policy_config.dt
        self._learn_per_step = policy_config.learn_per_step
        self._steps_btw_train = policy_config.steps_btw_train
        self._steps_btw_catchup = policy_config.steps_btw_catchup
        self._sampler = setup_memory(policy_config)
        self._count = 0
        self._learn_count = 0
        # optimization/storing
        self._device = device
        self._optimizer = setup_optimizer(
            self._qnet_function.parameters(),
            opt_name=policy_config.optimizer,
            lr=policy_config.lr, dt=self._dt,
            inverse_gradient_magnitude=1,
            weight_decay=policy_config.weight_decay)

        # scheduling parameters
        self._schedule_params = dict(
            mode='max', factor=.5, patience=25)
        self._schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, **self._schedule_params)
        # self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, policy_config.lr_decay)
        # logging
        self._cum_loss = 0
        self._log = 100


    def reset(self):
        # internals
        self._obs = np.array([])
        self._action = np.array([])
        self._reward = np.array([])
        self._next_obs = np.array([])
        self._done = np.array([])

    def step(self, obs: Arrayable):
        if self._train:
            self._obs = obs

        action = self.act(obs)
        if self._train:
            self._action = action

        return action

    def observe(self,
                next_obs: Arrayable,
                reward: Arrayable,
                done: Arrayable):
        if self._train:
            self._count += 1
            self._next_obs = next_obs
            self._reward = reward
            self._done = done
            self._sampler.push(
                self._obs, self._action, self._next_obs, self._reward, self._done)
            self.learn()


    def act(self, obs: Arrayable):
        with torch.no_grad():
            pre_action = self._qnet_noise.perturb_output(
                obs, function=self._qnet_function)
            self._qnet_noise.step()
            return pre_action.argmax(axis=-1)

    def learn(self):
        if self._count % self._steps_btw_train == self._steps_btw_train - 1:
            # TODO: enlever try except
            try:
                for _ in range(self._learn_per_step):
                    obs, action, next_obs, reward, done, weights = self._sampler.sample()
                    reward = arr_to_th(reward, self._device)
                    weights = arr_to_th(check_array(weights), self._device)
                    # for now, discrete actions
                    indices = arr_to_th(action, self._device).long()

                    q = self._qnet_function(obs).gather(1, indices.view(-1, 1)).squeeze()
                    target = torch.max(self._target_function(next_obs), dim=1)[0].squeeze()

                    exp_q = reward + self._gamma ** self._dt 
                    if self._gamma == 1.:
                        assert (1 - done).all(), "Gamma set to 1. with a potentially episodic problem..."
                        exp_q += target.detach()
                    else:
                        done = arr_to_th(done.astype('float'), self._device)
                        exp_q += self._gamma ** self._dt * target#.detach()

                    loss = (((q - exp_q) ** 2) * weights).mean()
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                    self._cum_loss += loss.item()
                    self._learn_count += 1

                log("Avg_adv_loss", self._cum_loss / self._learn_count, self._learn_count)
                info(f'At iteration {self._learn_count}, avg_loss: {self._cum_loss/self._learn_count}')

            except IndexError:
                # If not enough data in the buffer, do nothing
                pass

        if self._count % self._steps_btw_catchup == self._steps_btw_catchup -1:
            self._target_function.load_state_dict(self._qnet_function.state_dict())


    def train(self):
        self._train = True
        self._qnet_function.train()

    def eval(self):
        self._train = False
        self._qnet_function.eval()

    def observe_evaluation(self, eval_return: float):
        self._schedulers.step(eval_return)

    def value(self, obs: Arrayable):
        return th_to_arr(torch.max(self._qnet_function(obs), dim=1)[0])

    def advantage(self, obs: Arrayable):
        q = self._qnet_function(obs)
        return th_to_arr(q - torch.max(q, dim=1))

    def load_state_dict(self, state_dict: StateDict):
        self._optimizers.load_state_dict(state_dict['optimizer'])
        self._qnet_function.load_state_dict(state_dict['qnet_function'])
        self._target_function.load_state_dict(state_dict['target_function'])
        self._schedulers.load_state_dict(state_dict['schedulers'])
        # self._schedulers.last_epoch = state_dict['iteration']

    def state_dict(self):
        return {
            "optimizer": self._optimizer.state_dict(),
            "qnet_function": self._qnet_function.state_dict(),
            "target_function": self._target_function.state_dict(),
            "schedulers": self._schedulers.state_dict(),
            "iteration": self._schedulers.last_epoch}
