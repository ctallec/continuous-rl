"""Define continuous policy."""
from itertools import chain
import torch
from torch import Tensor
import numpy as np
from abstract import ParametricFunction, Arrayable, Noise, StateDict
from convert import th_to_arr, check_array
from config import SampledAdvantagePolicyConfig, AdvantagePolicyConfig
from policies.shared import SharedAdvantagePolicy
from logging import info
from mylog import log

class SampledAdvantagePolicy(SharedAdvantagePolicy):
    def __init__(self,
                 adv_function: ParametricFunction,
                 val_function: ParametricFunction,
                 adv_noise: Noise,
                 policy_config: SampledAdvantagePolicyConfig,
                 action_shape,
                 device) -> None:
        super().__init__(policy_config, val_function, device)

        self._adv_function = adv_function.to(device)
        self._adv_noise = adv_noise
        self._nb_samples = policy_config.nb_samples
        self._action_shape = action_shape

        # optimization/storing
        self._optimizers = (
            torch.optim.SGD(chain(self._adv_function.parameters(),
                                  [self._baseline]), lr=policy_config.lr * self._dt),
            torch.optim.SGD(self._val_function.parameters(),
                            lr=policy_config.lr * self._dt ** 2))
        self._schedulers = (
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[0],
                                              policy_config.lr_decay),
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[1],
                                              policy_config.lr_decay))

        # logging
        self._cum_loss = 0

    def act(self, obs: Arrayable):
        with torch.no_grad():
            obs = check_array(obs)
            batch_size = obs.shape[0]
            o_dim = obs.ndim

            proposed_actions = np.random.uniform(
                -1, 1, [self._nb_samples, batch_size, *self._action_shape])
            advantages = self._adv_noise.perturb_output(
                np.tile(obs, [self._nb_samples] + o_dim * [1]),
                proposed_actions, function=self._adv_function).squeeze()
            self._adv_noise.step()
            action_idx = np.argmax(advantages, axis=0)
            action = proposed_actions[action_idx, np.arange(obs.shape[0])]
        return action

    def compute_advantages(self, obs: Arrayable, action: Arrayable) -> Tensor:
        obs = check_array(obs)
        action = check_array(action)
        proposed_actions = np.random.uniform(
            -1, 1, [self._nb_samples, obs.shape[0], *self._action_shape])
        adv = self._adv_function(obs, action).squeeze()
        max_adv = torch.max(
            self._adv_function(
                np.tile(obs, [self._nb_samples] + obs.ndim * [1]),
                proposed_actions).squeeze(), dim=0)[0]
        return adv, max_adv

    def optimize_value(self, *losses: Tensor):
        self._optimizers[0].zero_grad()
        self._optimizers[1].zero_grad()
        (losses[0] + losses[1]).backward()
        self._optimizers[0].step()
        self._schedulers[0].step()
        self._optimizers[1].step()
        self._schedulers[1].step()

        # logging
        self._cum_loss += losses[0].item()
        self._learn_count += 1

    def optimize_policy(self, max_adv: Tensor):
        pass

    def log(self):
        log("Avg_adv_loss", self._cum_loss / self._learn_count,
            self._learn_count)
        info(f"At iteration {self._learn_count}, "
             f"avg_loss: {self._cum_loss/self._learn_count}")

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

    def advantage(self, obs: Arrayable, action: Arrayable):
        return th_to_arr(self._adv_function(obs, action))

    def load_state_dict(self, state_dict: StateDict):
        self._optimizers[0].load_state_dict(state_dict['advantage_optimizer'])
        self._optimizers[1].load_state_dict(state_dict['value_optimizer'])
        self._adv_function.load_state_dict(state_dict['adv_function'])
        self._val_function.load_state_dict(state_dict['val_function'])
        self._schedulers[0].last_epoch = state_dict['iteration']
        self._schedulers[1].last_epoch = state_dict['iteration']

    def state_dict(self) -> StateDict:
        return {
            "advantage_optimizer": self._optimizers[0].state_dict(),
            "value_optimizer": self._optimizers[1].state_dict(),
            "adv_function": self._adv_function.state_dict(),
            "val_function": self._val_function.state_dict(),
            "iteration": self._schedulers[0].last_epoch}

class AdvantagePolicy(SharedAdvantagePolicy):
    def __init__(self,
                 adv_function: ParametricFunction,
                 val_function: ParametricFunction,
                 policy_function: ParametricFunction,
                 policy_noise: Noise,
                 policy_config: AdvantagePolicyConfig,
                 device) -> None:
        super().__init__(policy_config, val_function, device)

        self._adv_function = adv_function.to(device)
        self._policy_function = policy_function.to(device)
        self._policy_noise = policy_noise

        # TODO: I think we could optimize by gathering policy and advantage parameters
        self._optimizers = (
            torch.optim.SGD(chain(self._adv_function.parameters(), [self._baseline]),
                            lr=policy_config.lr * self._dt),
            torch.optim.SGD(self._val_function.parameters(),
                            lr=policy_config.lr * self._dt ** 2),
            torch.optim.SGD(self._policy_function.parameters(),
                            lr=policy_config.lr * self._dt))
        self._schedulers = (
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[0],
                                              policy_config.lr_decay),
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[1],
                                              policy_config.lr_decay),
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[2],
                                              policy_config.lr_decay))

        # logging
        self._cum_loss = 0
        self._cum_policy_loss = 0

    def act(self, obs: Arrayable):
        with torch.no_grad():
            action = self._policy_noise.perturb_output(
                obs, function=self._policy_function)
            self._policy_noise.step()
        return action

    def compute_advantages(self, obs: Arrayable, action: Arrayable) -> Tensor:
        adv = self._adv_function(obs, action).squeeze()
        max_adv = self._adv_function(obs, self._policy_function(obs)).squeeze()
        return adv, max_adv

    def optimize_value(self, *losses: Tensor):
        assert len(losses) == 2
        self._optimizers[0].zero_grad()
        self._optimizers[1].zero_grad()
        (losses[0] + losses[1]).backward(retain_graph=True)
        self._optimizers[0].step()
        self._schedulers[0].step()
        self._optimizers[1].step()
        self._schedulers[1].step()

        # logging
        self._cum_loss += losses[0].item()
        self._learn_count += 1

    def optimize_policy(self, max_adv: Tensor):
        policy_loss = - max_adv.mean()
        self._optimizers[2].zero_grad()
        policy_loss.backward()
        self._optimizers[2].step()
        self._schedulers[2].step()

        # logging
        self._cum_policy_loss += policy_loss.item()

    def log(self):
        info(f'At iteration {self._learn_count}, '
             f'Avg_adv_loss: {self._cum_loss/self._learn_count}, '
             f'Avg_policy_loss: {self._cum_policy_loss / self._learn_count}')
        log("Avg_adv_loss", self._cum_loss / self._learn_count,
            self._learn_count)
        log("Avg_policy_loss", self._cum_policy_loss / self._learn_count,
            self._learn_count)

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

    def load_state_dict(self, state_dict: StateDict):
        self._optimizers[0].load_state_dict(state_dict['advantage_optimizer'])
        self._optimizers[1].load_state_dict(state_dict['value_optimizer'])
        self._optimizers[2].load_state_dict(state_dict['policy_optimizer'])
        self._adv_function.load_state_dict(state_dict['adv_function'])
        self._val_function.load_state_dict(state_dict['val_function'])
        self._policy_function.load_state_dict(state_dict['policy_function'])
        self._schedulers[0].last_epoch = state_dict['iteration']
        self._schedulers[1].last_epoch = state_dict['iteration']

    def state_dict(self) -> StateDict:
        return {
            "advantage_optimizer": self._optimizers[0].state_dict(),
            "value_optimizer": self._optimizers[1].state_dict(),
            "policy_optimizer": self._optimizers[2].state_dict(),
            "adv_function": self._adv_function.state_dict(),
            "val_function": self._val_function.state_dict(),
            "policy_function": self._policy_function.state_dict(),
            "iteration": self._schedulers[0].last_epoch}
