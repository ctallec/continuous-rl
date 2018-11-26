"""Define policies."""
from typing import Optional
import torch
from torch import Tensor
from abstract import ParametricFunction, Arrayable, Noise, StateDict
from convert import arr_to_th, th_to_arr
from config import PolicyConfig, AdvantagePolicyConfig
from policies.shared import SharedAdvantagePolicy
from mylog import log
from logging import info
from optimizer import setup_optimizer

class AdvantagePolicy(SharedAdvantagePolicy):
    def __init__(self,
                 adv_function: ParametricFunction,
                 val_function: ParametricFunction,
                 adv_noise: Noise,
                 policy_config: PolicyConfig,
                 device) -> None:
        super().__init__(policy_config, val_function, device) # type: ignore
        assert isinstance(policy_config, AdvantagePolicyConfig)
        self._adv_function = adv_function.to(device)
        self._adv_noise = adv_noise

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
        self._float_loss: Optional[float] = None
        self._log = 100

    def act(self, obs: Arrayable):
        with torch.no_grad():
            pre_action = self._adv_noise.perturb_output(
                obs, function=self._adv_function)
            self._adv_noise.step()
            return pre_action.argmax(axis=-1)

    def compute_advantages(self, obs: Arrayable, action: Arrayable) -> Tensor:
        indices = arr_to_th(action, self._device).long()
        adv_a = self._adv_function(obs)
        max_adv = torch.max(adv_a, dim=1)[0]
        adv = adv_a.gather(1, indices.view(-1, 1)).squeeze()
        return adv, max_adv

    def optimize_value(self, *losses: Tensor):
        assert len(losses) == 1
        self._optimizers[0].zero_grad()
        self._optimizers[1].zero_grad()
        losses[0].backward()
        self._optimizers[0].step()
        self._optimizers[1].step()

        # logging
        self._cum_loss += losses[0].item()
        if self._float_loss is None:
            self._float_loss = losses[0].item()
        else:
            self._float_loss = self._float_loss * self._gamma ** self._dt +\
                (1 - self._gamma ** self._dt) * losses[0].item()
        self._learn_count += 1

    def optimize_policy(self, max_adv: Tensor):
        pass

    def observe_evaluation(self, eval_return: float):
        self._schedulers[0].step(eval_return)
        self._schedulers[1].step(eval_return)

    def log(self):
        log("Avg_adv_loss", self._cum_loss / self._learn_count, self._learn_count)
        log("Float_adv_loss", self._float_loss, self._learn_count)
        info(f"At iteration {self._learn_count}, "
             f'Float_adv_loss: {self._float_loss}, '
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

    def advantage(self, obs: Arrayable):
        return th_to_arr(self._adv_function(obs))

    def load_state_dict(self, state_dict: StateDict):
        self._optimizers[0].load_state_dict(state_dict['advantage_optimizer'])
        self._optimizers[1].load_state_dict(state_dict['value_optimizer'])
        self._adv_function.load_state_dict(state_dict['adv_function'])
        self._val_function.load_state_dict(state_dict['val_function'])
        self._schedulers[0].load_state_dict(state_dict['advantage_scheduler'])
        self._schedulers[1].load_state_dict(state_dict['value_scheduler'])

    def state_dict(self):
        return {
            "advantage_optimizer": self._optimizers[0].state_dict(),
            "value_optimizer": self._optimizers[1].state_dict(),
            "adv_function": self._adv_function.state_dict(),
            "val_function": self._val_function.state_dict(),
            "advantage_scheduler": self._schedulers[0].state_dict(),
            "value_scheduler": self._schedulers[1].state_dict(),
            "iteration": self._schedulers[0].last_epoch}

    def networks(self):
        return self._adv_function, self._val_function
