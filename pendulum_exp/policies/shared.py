""" Define shared elements between continuous and discrete. """
import numpy as np
from abstract import Policy, Arrayable, ParametricFunction, Actor, StateDict
from memory.utils import setup_memory
import torch
from torch import Tensor
from convert import arr_to_th, check_array, th_to_arr
from typing import Optional
from mylog import log
from stateful import CompoundStateful
from optimizer import setup_optimizer

class AdvantagePolicy(Policy, CompoundStateful):
    def __init__(self, gamma: float, dt: float, lr: float, optimizer: str,
                 learn_per_step: int, steps_btw_train: int, memory_size: int,
                 batch_size: int, alpha: Optional[float], beta: Optional[float],
                 val_function: ParametricFunction, adv_function: ParametricFunction,
                 actor: Actor, device) -> None:
        self._train = True
        self.reset()

        # parameters
        self._gamma = gamma
        self._dt = dt
        self._learn_per_step = learn_per_step
        self._steps_btw_train = steps_btw_train
        self._count = 0
        self._learn_count = 0
        self._device = device

        # logging
        self._log_step = 0
        self._stats_obs = None
        self._stats_actions = None

        # learning
        self._adv_function = adv_function.to(device)
        self._val_function = val_function.to(device)
        self._actor = actor.to(device)
        self._sampler = setup_memory(
            alpha=alpha, beta=beta, memory_size=memory_size, batch_size=batch_size)

        # optimization/storing
        self._adv_optimizer = \
            setup_optimizer(self._adv_function.parameters(),
                            opt_name=optimizer, lr=lr, dt=dt,
                            inverse_gradient_magnitude=1,
                            weight_decay=0)
        self._val_optimizer = \
            setup_optimizer(self._val_function.parameters(),
                            opt_name=optimizer, lr=lr, dt=dt,
                            inverse_gradient_magnitude=dt,
                            weight_decay=0)

    def reset(self):
        # internals
        self._obs = np.array([])
        self._action = np.array([])
        self._reward = np.array([])
        self._next_obs = np.array([])
        self._done = np.array([])
        self._time_limit = np.array([])

    def act(self, obs: Arrayable):
        return self._actor.act_noisy(obs)

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
                time_limit: Optional[Arrayable] = None):
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
                    self._actor.optimize(-max_adv.mean())

                self.log()
                self.log_stats()
            except IndexError as e:
                # If not enough data in the buffer, do nothing
                raise e
                pass

    def reset_log(self):
        self._cum_loss = 0
        self._log_step = 0

    def compute_advantages(self, obs: Arrayable, action: Arrayable) -> Tensor:
        """Computes adv, max_adv."""
        max_action = self.actor.act(obs)
        if len(self._adv_function.input_shape()) == 2:
            adv = self._adv_function(obs, action).squeeze()
            max_adv = self._adv_function(obs, max_action).squeeze()
        else:
            adv_all = self._adv_function(obs)
            action = arr_to_th(action, self._device).long()
            adv = adv_all.gather(1, action.view(-1, 1)).squeeze()
            max_adv = adv_all.gather(1, max_action.view(-1, 1)).squeeze()
        return adv, max_adv

    def compute_next_value(self, next_obs: Arrayable, done: Tensor, mean_v: Tensor) -> Tensor:
        """Also detach next value."""
        next_v = self._val_function(next_obs).squeeze().detach()
        if self._gamma == 1:
            assert (1 - done).byte().all(), "Gamma set to 1. with a potentially"\
                "episodic problem..."
            return next_v
        return (1 - done) * next_v - done * mean_v * self._gamma / (1 - self._gamma)

    def optimize_value(self, *losses: Tensor):
        assert len(losses) == 1
        self._adv_optimizer.zero_grad()
        self._val_optimizer.zero_grad()
        losses[0].mean().backward()
        self._adv_optimizer.step()
        self._val_optimizer.step()

        # logging
        self._cum_loss += losses[0].sqrt().mean().item()
        self._log_step += 1
        self._learn_count += 1

    def log_stats(self):
        if self._stats_obs is None:
            self._stats_obs, self._stats_actions, _, _, _, _, _ = self._sampler.sample(to_observe=False)

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

    def state_dict(self) -> StateDict:
        state = super(CompoundStateful, self).state_dict()
        state["learn_count"] = self._learn_count
        return state

    def load_state_dcit(self, state_dict: StateDict):
        state = super(CompoundStateful, self).load_state_dict(state_dict)
        self._learn_count = state["learn_count"]
