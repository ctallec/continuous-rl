"""Define policies."""
import torch
import numpy as np
from abstract import Policy, ParametricFunction, Arrayable, Noise
from convert import arr_to_th, th_to_arr
from stats import FloatingAvg, penalize_mean
from typing import Callable

class AdvantagePolicy(Policy):
    def __init__(self,
                 adv_function: ParametricFunction,
                 val_function: ParametricFunction,
                 adv_noise: Noise,
                 gamma: float,
                 avg_alpha: float,
                 dt: float,
                 lr: float,
                 lr_decay: Callable[[int], float],
                 device) -> None:
        self._adv_function = adv_function.to(device)
        self._val_function = val_function.to(device)
        self._adv_noise = adv_noise

        # optimization/storing
        self._device = device
        self._optimizers = (
            torch.optim.SGD(adv_function.parameters(), lr=lr * dt),
            torch.optim.SGD(val_function.parameters(), lr=lr * dt ** 2))
        self._schedulers = (
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[0], lr_decay),
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[1], lr_decay))

        # parameters
        self._gamma = gamma
        self._dt = dt

        # internals
        self._obs = np.array([])
        self._action = np.array([])
        self._reward = np.array([])
        self._next_obs = np.array([])
        self._done = np.array([])

        self._reward_avg = FloatingAvg(avg_alpha * self._dt)

        self._train = True

        # logging
        self._cum_loss = 0
        self._count = 0
        self._log = 100

    def step(self, obs: Arrayable):
        if self._train:
            self._obs = obs

        with torch.no_grad():
            pre_action = self._adv_noise.perturb_output(
                obs, self._adv_function)
            self._adv_noise.step()
            action = pre_action.argmax(axis=-1)
            if self._train:
                self._action = action

        return action

    def observe(self,
                next_obs: Arrayable,
                reward: Arrayable,
                done: Arrayable):
        if self._train:
            self._next_obs = next_obs
            self._reward = reward
            self._done = done
            self.learn()

    def learn(self):
        if self._obs.shape == self._next_obs.shape:
            # for now, discrete actions
            indices = arr_to_th(self._action, self._device).long()
            v = self._val_function(self._obs).squeeze()
            adv = self._adv_function(self._obs)
            max_adv = torch.max(adv, dim=1)[0]
            adv_a = adv.gather(1, indices.view(-1, 1)).squeeze()

            discounted_next_v = self._gamma ** self._dt * self._val_function(
                self._next_obs).squeeze()
            expected_v = arr_to_th((self._reward - self._reward_avg.mean), self._device) * self._dt + \
                discounted_next_v.detach()
            dv = (expected_v - v) / self._dt
            self._reward_avg.step(self._reward)
            a_update = dv - adv_a + max_adv

            adv_update_loss = (a_update ** 2).mean()
            adv_norm_loss = (max_adv ** 2).mean()
            mean_loss = penalize_mean(v)
            loss = adv_update_loss + adv_norm_loss

            self._optimizers[0].zero_grad()
            self._optimizers[1].zero_grad()
            (mean_loss / self._dt + loss).backward()
            self._optimizers[0].step()
            self._schedulers[0].step()
            self._optimizers[1].step()
            self._schedulers[1].step()

            # logging
            self._cum_loss += loss.item()
            if self._count % self._log == self._log - 1:
                print(f'At iteration {self._count}, avg_loss: {self._cum_loss/self._count}')

            self._count += 1

    def train(self):
        self._train = True
        self._val_function.train()
        self._adv_function.train()

    def eval(self):
        self._train = False
        self._val_function.eval()
        self._adv_function.eval()

    def value(self, obs: Arrayable):
        return th_to_arr(self._val_function(obs))

    def advantage(self, obs: Arrayable):
        return th_to_arr(self._adv_function(obs))

    def reset(self):
        # internals
        self._obs = np.array([])
        self._action = np.array([])
        self._reward = np.array([])
        self._next_obs = np.array([])
        self._done = np.array([])
