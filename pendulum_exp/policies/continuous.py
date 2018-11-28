"""Define continuous policy."""
import torch
from torch import Tensor
import numpy as np
from abstract import ParametricFunction, Arrayable, Noise, StateDict
from convert import th_to_arr, check_array
from config import SampledAdvantagePolicyConfig, ApproximateAdvantagePolicyConfig
from policies.shared import SharedAdvantagePolicy
from logging import info
from mylog import log
from optimizer import setup_optimizer

class SampledAdvantagePolicy(SharedAdvantagePolicy):
    def __init__(self,
                 adv_function: ParametricFunction,
                 val_function: ParametricFunction,
                 adv_noise: Noise,
                 policy_config: SampledAdvantagePolicyConfig,
                 action_shape,
                 device) -> None:
        super().__init__(policy_config, val_function, device) # type: ignore

        self._adv_function = adv_function.to(device)
        self._adv_noise = adv_noise
        self._nb_samples = policy_config.nb_samples
        self._action_shape = action_shape

        # optimization/storing
        self._optimizers = (
            setup_optimizer(self._adv_function.parameters(),
                            opt_name=policy_config.optimizer,
                            lr=policy_config.lr, dt=self._dt,
                            inverse_gradient_magnitude=1,
                            weight_decay=policy_config.weight_decay),
            setup_optimizer(self._val_function.parameters(),
                            opt_name=policy_config.optimizer,
                            lr=policy_config.lr, dt=self._dt,
                            inverse_gradient_magnitude=self._dt,
                            weight_decay=policy_config.weight_decay))

        self._schedulers = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizers[0], **self._schedule_params),
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizers[1], **self._schedule_params))

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
        losses[0].mean().backward()
        self._optimizers[0].step()
        self._optimizers[1].step()

        # logging
        self._cum_loss += losses[0].sqrt().mean().item()
        self._log_step += 1
        self._learn_count += 1

    def optimize_policy(self, max_adv: Tensor):
        pass

    def reset_log(self):
        self._cum_loss = 0
        self._log_step = 0

    def log(self):
        log("loss/advantage", self._cum_loss / self._log_step, self._learn_count)
        info(f"At iteration {self._learn_count}, "
             f'adv_loss: {self._cum_loss/self._log_step}')

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

    def observe_evaluation(self, eval_return: float):
        self._schedulers[0].step(eval_return)
        self._schedulers[1].step(eval_return)

    def load_state_dict(self, state_dict: StateDict):
        self._optimizers[0].load_state_dict(state_dict['advantage_optimizer'])
        self._optimizers[1].load_state_dict(state_dict['value_optimizer'])
        self._adv_function.load_state_dict(state_dict['adv_function'])
        self._val_function.load_state_dict(state_dict['val_function'])
        self._schedulers[0].load_state_dict(state_dict['advantage_scheduler'])
        self._schedulers[1].load_state_dict(state_dict['value_scheduler'])
        self._learn_count = state_dict['learn_count']

    def state_dict(self) -> StateDict:
        return {
            "advantage_optimizer": self._optimizers[0].state_dict(),
            "value_optimizer": self._optimizers[1].state_dict(),
            "adv_function": self._adv_function.state_dict(),
            "val_function": self._val_function.state_dict(),
            "advantage_scheduler": self._schedulers[0].state_dict(),
            "value_scheduler": self._schedulers[1].state_dict(),
            "iteration": self._schedulers[0].last_epoch,
            "learn_count": self._learn_count
        }

    def networks(self):
        return self._adv_function, self._val_function

    def _get_stats(self):
        obs = check_array(self._stats_obs)
        batch_size = obs.shape[0]
        o_dim = obs.ndim

        proposed_actions = np.random.uniform(
            -1, 1, [self._nb_samples, batch_size, *self._action_shape])
        advantages = self._adv_function(
            np.tile(obs, [self._nb_samples] + o_dim * [1]),
            proposed_actions).squeeze()
        action_idx = np.argmax(advantages, axis=0)
        actions = proposed_actions[action_idx, np.arange(obs.shape[0])].astype('float32')
        V = self._val_function(self._stats_obs).squeeze().cpu().numpy()
        return V, actions

class AdvantagePolicy(SharedAdvantagePolicy):
    def __init__(self,
                 adv_function: ParametricFunction,
                 val_function: ParametricFunction,
                 policy_function: ParametricFunction,
                 policy_noise: Noise,
                 policy_config: ApproximateAdvantagePolicyConfig,
                 device) -> None:
        super().__init__(policy_config, val_function, device) # type: ignore

        self._adv_function = adv_function.to(device)
        self._policy_function = policy_function.to(device)
        self._policy_noise = policy_noise

        # TODO: I think we could optimize by gathering policy and advantage parameters
        self._optimizers = (
            setup_optimizer(self._adv_function.parameters(),
                            opt_name=policy_config.optimizer,
                            lr=policy_config.lr, dt=self._dt,
                            inverse_gradient_magnitude=1,
                            weight_decay=policy_config.weight_decay),
            setup_optimizer(self._val_function.parameters(),
                            opt_name=policy_config.optimizer,
                            lr=policy_config.lr, dt=self._dt,
                            inverse_gradient_magnitude=self._dt,
                            weight_decay=policy_config.weight_decay),
            setup_optimizer(self._policy_function.parameters(),
                            opt_name=policy_config.optimizer,
                            lr=policy_config.policy_lr, dt=self._dt,
                            inverse_gradient_magnitude=1,
                            weight_decay=policy_config.weight_decay))

        self._schedulers = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizers[0], **self._schedule_params),
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizers[1], **self._schedule_params),
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizers[2], **self._schedule_params)
        )

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
        self._optimizers[0].zero_grad()
        self._optimizers[1].zero_grad()
        losses[0].mean().backward(retain_graph=True)
        self._optimizers[0].step()
        self._optimizers[1].step()

        # logging
        self._cum_loss += losses[0].sqrt().mean().item()
        self._log_step += 1
        self._learn_count += 1

    def optimize_policy(self, max_adv: Tensor):
        policy_loss = - max_adv.mean()
        self._optimizers[2].zero_grad()
        policy_loss.backward()
        self._optimizers[2].step()

        # logging
        self._cum_policy_loss += policy_loss.item()

    def reset_log(self):
        self._log_step = 0
        self._cum_loss = 0
        self._cum_policy_loss = 0

    def log(self):
        info(f'At iteration {self._learn_count}, '
             f'adv_loss: {self._cum_loss/self._log_step}, '
             f'policy_loss: {self._cum_policy_loss / self._log_step}')
        log("loss/advantage", self._cum_loss / self._log_step, self._learn_count)
        log("loss/policy", self._cum_policy_loss / self._log_step, self._learn_count)

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

    def observe_evaluation(self, eval_return: float):
        self._schedulers[0].step(eval_return)
        self._schedulers[1].step(eval_return)
        self._schedulers[2].step(eval_return)

    def advantage(self, obs: Arrayable, action: Arrayable):
        return th_to_arr(self._adv_function(obs, action))

    def load_state_dict(self, state_dict: StateDict):
        self._optimizers[0].load_state_dict(state_dict['advantage_optimizer'])
        self._optimizers[1].load_state_dict(state_dict['value_optimizer'])
        self._optimizers[2].load_state_dict(state_dict['policy_optimizer'])
        self._adv_function.load_state_dict(state_dict['adv_function'])
        self._val_function.load_state_dict(state_dict['val_function'])
        self._policy_function.load_state_dict(state_dict['policy_function'])
        self._schedulers[0].load_state_dict(state_dict['advantage_scheduler'])
        self._schedulers[1].load_state_dict(state_dict['value_scheduler'])
        self._schedulers[2].load_state_dict(state_dict['policy_scheduler'])
        self._learn_count = state_dict['learn_count']

    def state_dict(self) -> StateDict:
        return {
            "advantage_optimizer": self._optimizers[0].state_dict(),
            "value_optimizer": self._optimizers[1].state_dict(),
            "policy_optimizer": self._optimizers[2].state_dict(),
            "adv_function": self._adv_function.state_dict(),
            "val_function": self._val_function.state_dict(),
            "policy_function": self._policy_function.state_dict(),
            "advantage_scheduler": self._schedulers[0].state_dict(),
            "value_scheduler": self._schedulers[1].state_dict(),
            "policy_scheduler": self._schedulers[2].state_dict(),
            "iteration": self._schedulers[0].last_epoch,
            "learn_count": self._learn_count
        }

    def networks(self):
        return self._adv_function, self._val_function, self._policy_function

    def _get_stats(self):
        V = self._val_function(self._stats_obs).squeeze().cpu().numpy()
        actions = self._policy_function(self._stats_obs).cpu().numpy()
        return V, actions
