"""Define continuous policy."""
from itertools import chain
import torch
import numpy as np
from abstract import Policy, ParametricFunction, Arrayable, Noise
from convert import arr_to_th, th_to_arr
from stats import penalize_mean
from typing import Callable

class AdvantagePolicy(Policy):
    def __init__(self,
                 adv_function: ParametricFunction,
                 val_function: ParametricFunction,
                 policy_function: ParametricFunction,
                 policy_noise: Noise,
                 alpha: float,
                 gamma: float,
                 dt: float,
                 lr: float,
                 lr_decay: Callable[[int], float],
                 device) -> None:
        self._alpha = alpha
        self._adv_function = adv_function.to(device)
        self._val_function = val_function.to(device)
        self._baseline = torch.nn.Parameter(torch.Tensor([0.])).to(device)
        self._policy_function = policy_function.to(device)
        self._policy_noise = policy_noise

        # optimization/storing
        self._device = device
        # TODO: I think we could optimize by gathering policy and advantage parameters
        self._optimizers = (
            torch.optim.SGD(chain(self._adv_function.parameters(), [self._baseline]), lr=lr * dt),
            torch.optim.SGD(self._val_function.parameters(), lr=lr * dt ** 2),
            torch.optim.SGD(self._policy_function.parameters(), lr=2 * lr * dt))
        self._schedulers = (
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[0], lr_decay),
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[1], lr_decay),
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[2], lr_decay))

        # parameters
        self._gamma = gamma
        self._dt = dt

        # internals
        self._obs = np.array([])
        self._action = np.array([])
        self._reward = np.array([])
        self._next_obs = np.array([])
        self._done = np.array([])

        self._train = True

        # logging
        self._cum_loss = 0
        self._cum_policy_loss = 0
        self._count = 0
        self._log = 100

    def step(self, obs: Arrayable):
        if self._train:
            self._obs = obs

        with torch.no_grad():
            action = self._policy_noise.perturb_output(
                obs, self._policy_function)
            self._policy_noise.step()
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
            v = self._val_function(self._obs).squeeze()
            adv_a = self._adv_function(self._obs, self._action).squeeze()
            max_adv = self._adv_function(self._obs, self._policy_function(self._obs)).squeeze()

            if self._gamma == 1:
                assert (1 - self._done).all(), "Gamma set to 1. with a potentially episodic problem..."
                discounted_next_v = self._gamma ** self._dt * self._val_function(self._next_obs).squeeze()
            else:
                done = arr_to_th(self._done.astype('float'), self._device)
                discounted_next_v = \
                    (1 - done) * self._gamma ** self._dt * self._val_function(self._next_obs).squeeze() -\
                    done * self._gamma ** self._dt * self._baseline / (1 - self._gamma)

            expected_v = (arr_to_th(self._reward, self._device) - self._baseline) * self._dt + \
                discounted_next_v.detach()
            dv = (expected_v - v) / self._dt
            a_update = dv - adv_a + max_adv

            adv_update_loss = (a_update ** 2).mean()
            adv_norm_loss = (max_adv ** 2).mean()
            mean_loss = self._alpha * penalize_mean(v)
            loss = adv_update_loss + adv_norm_loss

            self._optimizers[0].zero_grad()
            self._optimizers[1].zero_grad()
            (mean_loss / self._dt + loss).backward(retain_graph=True)
            self._optimizers[0].step()
            self._schedulers[0].step()
            self._optimizers[1].step()
            self._schedulers[1].step()

            policy_loss = - max_adv.mean()
            self._optimizers[2].zero_grad()
            policy_loss.backward()
            self._optimizers[2].step()
            self._schedulers[2].step()


            # logging
            self._cum_loss += loss.item()
            self._cum_policy_loss += policy_loss.item()
            if self._count % self._log == self._log - 1:
                print(f'At iteration {self._count}, avg_loss: {self._cum_loss/self._count}, '
                      f'avg_policy_loss: {self._cum_policy_loss / self._count}')

            self._count += 1

    def train(self):
        self._train = True
        self._val_function.train()
        self._adv_function.train()
        self._policy_function.train()

    def eval(self):
        self._train = False
        self._val_function.eval()
        self._adv_function.eval()
        self._policy_function.eval()

    def value(self, obs: Arrayable):
        return th_to_arr(self._val_function(obs))

    def advantage(self, obs: Arrayable, action: Arrayable):
        return th_to_arr(self._adv_function(obs, action))

    def reset(self):
        # internals
        self._obs = np.array([])
        self._action = np.array([])
        self._reward = np.array([])
        self._next_obs = np.array([])
        self._done = np.array([])
