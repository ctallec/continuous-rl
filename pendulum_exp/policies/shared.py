""" Define shared elements between continuous and discrete. """
import numpy as np
from abstract import Policy, Arrayable, ParametricFunction
from config import PolicyConfig
from memory.utils import setup_memory
import torch
from torch import Tensor
from convert import arr_to_th, check_array, th_to_arr
from typing import Optional
from mylog import log

class SharedAdvantagePolicy(Policy):
    def __init__(self, policy_config: PolicyConfig,
                 val_function: ParametricFunction, device) -> None:
        self._train = True
        self.reset()

        # parameters
        self._gamma = policy_config.gamma
        self._dt = policy_config.dt
        self._learn_per_step = policy_config.learn_per_step
        self._steps_btw_train = policy_config.steps_btw_train
        self._sampler = setup_memory(policy_config)
        self._count = 0
        self._learn_count = 0
        self._device = device
        self._val_function = val_function

        # scheduling parameters
        self._schedule_params = dict(
            mode='max', factor=.5, patience=25)

        # logging
        self._log_step = 0
        self._stats_obs = None
        self._stats_actions = None

    def reset(self):
        # internals
        self._obs = np.array([])
        self._action = np.array([])
        self._reward = np.array([])
        self._next_obs = np.array([])
        self._done = np.array([])
        self._time_limit = np.array([])

    def act(self, obs: Arrayable):
        raise NotImplementedError

    def step(self, obs: Arrayable):
        for net in self.networks():
            net.eval() # make sure batch norm is in eval mode

        if self._train:
            self._obs = obs

        action = self.act(obs)
        if self._train:
            self._action = action

        return action

    def observe(self,
                next_obs: Arrayable,
                reward: Arrayable,
                done: Arrayable,
                time_limit: Optional[Arrayable]=None):
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
        for net in self.networks():
            net.train() # batch norm in train mode

        if self._count % self._steps_btw_train == self._steps_btw_train - 1:
            try:
                self.reset_log()
                for _ in range(self._learn_per_step):
                    obs, action, next_obs, reward, done, weights, time_limit = \
                        self._sampler.sample()

                    # don't update when a time limit is reached
                    if time_limit is not None:
                        weights = weights * (1 - time_limit)
                    reference_obs = self._sampler.reference_obs
                    reward = arr_to_th(reward, self._device)
                    weights = arr_to_th(check_array(weights), self._device)
                    done = arr_to_th(check_array(done).astype('float'), self._device)

                    v = self._val_function(obs).squeeze()
                    reference_v = self._val_function(reference_obs).squeeze().detach()
                    mean_v = reference_v.mean()
                    next_v = self.compute_next_value(next_obs, done, mean_v)
                    adv, max_adv = self.compute_advantages(
                        obs, action)

                    expected_v = reward * self._dt + \
                        self._gamma ** self._dt * next_v
                    dv = (expected_v - v) / self._dt - (1 - done) * self._gamma * mean_v
                    bell_residual = dv - adv + max_adv
                    self._sampler.observe(np.abs(th_to_arr(bell_residual)))

                    adv_update_loss = ((bell_residual ** 2) * weights)
                    adv_norm_loss = ((max_adv ** 2) * weights)
                    bell_loss = adv_update_loss + adv_norm_loss

                    self.optimize_value(bell_loss)
                    self.optimize_policy(max_adv)

                self.log()
                self.log_stats()
            except IndexError as e:
                # If not enough data in the buffer, do nothing
                raise e
                pass

    def optimize_value(self, *losses: Tensor):
        raise NotImplementedError()

    def optimize_policy(self, max_adv: Tensor):
        raise NotImplementedError()

    def compute_advantages(self, obs: Arrayable, action: Arrayable) -> Tensor:
        """Computes adv, max_adv."""
        raise NotImplementedError()

    def compute_next_value(self, next_obs: Arrayable, done: Tensor, mean_v: Tensor) -> Tensor:
        """Also detach next value."""
        next_v = self._val_function(next_obs).squeeze().detach()
        if self._gamma == 1:
            assert (1 - done).byte().all(), "Gamma set to 1. with a potentially"\
                "episodic problem..."
            return next_v
        return (1 - done) * next_v - done * mean_v * self._gamma / (1 - self._gamma)

    def log_stats(self):
        if self._stats_obs is None:
            self._stats_obs, self._stats_actions, _, _, _, _, _ = self._sampler.sample()

        with torch.no_grad():
            V, actions = self._get_stats()
            noisy_actions = self.act(self._stats_obs)
            adv, max_adv = self.compute_advantages(self._stats_obs, self._stats_actions)
            reference_v = self._val_function(self._sampler.reference_obs).squeeze().detach()
            log("stats/mean_v", V.mean().item(), self._learn_count)
            log("stats/std_v", V.std().item(), self._learn_count)
            log("stats/mean_actions", actions.mean().item(), self._learn_count)
            log("stats/std_actions", actions.std().item(), self._learn_count)
            log("stats/mean_noisy_actions", noisy_actions.mean().item(), self._learn_count)
            log("stats/std_noisy_actions", noisy_actions.std().item(), self._learn_count)
            log("stats/mean_advantage", adv.mean().item(), self._learn_count)
            log("stats/std_advantage", adv.std().item(), self._learn_count)
            log("stats/mean_reference_v", reference_v.mean().item(), self._learn_count)
            log("stats/std_reference_v", reference_v.std().item(), self._learn_count)
