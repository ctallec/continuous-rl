"""Define policies."""
from itertools import chain
import torch
from torch import Tensor
from abstract import ParametricFunction, Arrayable, Noise, StateDict
from convert import arr_to_th, th_to_arr
from config import PolicyConfig
from policies.shared import SharedAdvantagePolicy
from mylog import log
from logging import info

class AdvantagePolicy(SharedAdvantagePolicy):
    def __init__(self,
                 adv_function: ParametricFunction,
                 val_function: ParametricFunction,
                 adv_noise: Noise,
                 policy_config: PolicyConfig,
                 device) -> None:
        super().__init__(policy_config, val_function, device)
        self._adv_function = adv_function.to(device)
        self._adv_noise = adv_noise

        # optimization/storing
        self._optimizers = (
            torch.optim.SGD(chain(self._adv_function.parameters(), [self._baseline]),
                            lr=policy_config.lr * policy_config.dt),
            torch.optim.SGD(self._val_function.parameters(),
                            lr=policy_config.lr * policy_config.dt ** 2))
        self._schedulers = (
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[0],
                                              policy_config.lr_decay),
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[1],
                                              policy_config.lr_decay))
        # logging
        self._cum_loss = 0
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
        assert len(losses) == 2
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
        log("Avg_adv_loss", self._cum_loss / self._learn_count, self._learn_count)
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

    def advantage(self, obs: Arrayable):
        return th_to_arr(self._adv_function(obs))

    def load_state_dict(self, state_dict: StateDict):
        self._optimizers[0].load_state_dict(state_dict['advantage_optimizer'])
        self._optimizers[1].load_state_dict(state_dict['value_optimizer'])
        self._adv_function.load_state_dict(state_dict['adv_function'])
        self._val_function.load_state_dict(state_dict['val_function'])
        self._schedulers[0].last_epoch = state_dict['iteration']
        self._schedulers[1].last_epoch = state_dict['iteration']

    def state_dict(self):
        return {
            "advantage_optimizer": self._optimizers[0].state_dict(),
            "value_optimizer": self._optimizers[1].state_dict(),
            "adv_function": self._adv_function.state_dict(),
            "val_function": self._val_function.state_dict(),
            "iteration": self._schedulers[0].last_epoch}
