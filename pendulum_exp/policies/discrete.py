"""Define policies."""
from itertools import chain
import torch
from abstract import ParametricFunction, Arrayable, Noise, StateDict
from convert import arr_to_th, th_to_arr
from stats import penalize_mean
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
        super().__init__(policy_config)
        self._adv_function = adv_function.to(device)
        self._val_function = val_function.to(device)
        self._baseline = torch.nn.Parameter(torch.Tensor([0.]).to(device))
        self._adv_noise = adv_noise

        # optimization/storing
        self._device = device
        self._optimizers = (
            torch.optim.SGD(chain(self._adv_function.parameters(), [self._baseline]), lr=policy_config.lr * policy_config.dt),
            torch.optim.SGD(self._val_function.parameters(), lr=policy_config.lr * policy_config.dt ** 2))
        self._schedulers = (
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[0], policy_config.lr_decay),
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[1], policy_config.lr_decay))
        # logging
        self._cum_loss = 0
        self._log = 100

    def act(self, obs: Arrayable):
        with torch.no_grad():
            pre_action = self._adv_noise.perturb_output(
                obs, self._adv_function)
            self._adv_noise.step()
            return pre_action.argmax(axis=-1)

    def learn(self):
        if self._count % self._steps_btw_train == self._steps_btw_train - 1:
            try:
                for _ in range(self._learn_per_step):
                    obs, action, next_obs, reward, done = self._sampler.sample()
                    # for now, discrete actions
                    indices = arr_to_th(action, self._device).long()
                    v = self._val_function(obs).squeeze()
                    adv = self._adv_function(obs)
                    max_adv = torch.max(adv, dim=1)[0]
                    adv_a = adv.gather(1, indices.view(-1, 1)).squeeze()

                    if self._gamma == 1:
                        assert (1 - done).all(), "Gamma set to 1. with a potentially episodic problem..."
                        discounted_next_v = self._gamma ** self._dt * self._val_function(next_obs).squeeze().detach()
                    else:
                        done = arr_to_th(done.astype('float'), self._device)
                        discounted_next_v = \
                            (1 - done) * self._gamma ** self._dt * self._val_function(next_obs).squeeze().detach() -\
                            done * self._gamma ** self._dt * self._baseline / (1 - self._gamma)

                    expected_v = (arr_to_th(reward, self._device) - self._baseline) * self._dt + \
                        discounted_next_v
                    dv = (expected_v - v) / self._dt
                    a_update = dv - adv_a + max_adv

                    adv_update_loss = (a_update ** 2).mean()
                    adv_norm_loss = (max_adv ** 2).mean()
                    mean_loss = self._alpha * penalize_mean(v)
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
                    self._learn_count += 1
                log("Avg_adv_loss", self._cum_loss / self._count, self._count)
                info(f'At iteration {self._learn_count}, avg_loss: {self._cum_loss/self._learn_count}')

            except IndexError:
                # If not enough data in the buffer, do nothing
                pass

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
