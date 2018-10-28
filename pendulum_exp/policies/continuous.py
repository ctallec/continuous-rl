"""Define continuous policy."""
from itertools import chain
import torch
from abstract import ParametricFunction, Arrayable, Noise
from convert import arr_to_th, th_to_arr
from stats import penalize_mean
from config import PolicyConfig
from policies.shared import SharedAdvantagePolicy

class AdvantagePolicy(SharedAdvantagePolicy):
    def __init__(self,
                 adv_function: ParametricFunction,
                 val_function: ParametricFunction,
                 policy_function: ParametricFunction,
                 policy_noise: Noise,
                 policy_config: PolicyConfig,
                 device) -> None:
        super().__init__(policy_config)

        self._adv_function = adv_function.to(device)
        self._val_function = val_function.to(device)
        self._baseline = torch.nn.Parameter(torch.Tensor([0.])).to(device)
        self._policy_function = policy_function.to(device)
        self._policy_noise = policy_noise

        # optimization/storing
        self._device = device
        # TODO: I think we could optimize by gathering policy and advantage parameters
        self._optimizers = (
            torch.optim.SGD(chain(self._adv_function.parameters(), [self._baseline]), lr=policy_config.lr * self._dt),
            torch.optim.SGD(self._val_function.parameters(), lr=policy_config.lr * self._dt ** 2),
            torch.optim.SGD(self._policy_function.parameters(), lr=2 * policy_config.lr * self._dt))
        self._schedulers = (
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[0], policy_config.lr_decay),
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[1], policy_config.lr_decay),
            torch.optim.lr_scheduler.LambdaLR(self._optimizers[2], policy_config.lr_decay))

        # logging
        self._cum_loss = 0
        self._cum_policy_loss = 0
        self._count = 0
        self._log = 100

    def act(self, obs: Arrayable):
        with torch.no_grad():
            action = self._policy_noise.perturb_output(
                obs, self._policy_function)
            self._policy_noise.step()
        return action

    def learn(self):
        try:
            for _ in range(self._learn_per_step):
                obs, action, next_obs, reward, done = self._sampler.sample()
                v = self._val_function(obs).squeeze()
                adv_a = self._adv_function(obs, action).squeeze()
                max_adv = self._adv_function(obs, self._policy_function(obs)).squeeze()

                if self._gamma == 1:
                    assert (1 - done).all(), "Gamma set to 1. with a potentially episodic problem..."
                    discounted_next_v = self._gamma ** self._dt * self._val_function(next_obs).squeeze()
                else:
                    done = arr_to_th(done.astype('float'), self._device)
                    discounted_next_v = \
                        (1 - done) * self._gamma ** self._dt * self._val_function(next_obs).squeeze() -\
                        done * self._gamma ** self._dt * self._baseline / (1 - self._gamma)

                expected_v = (arr_to_th(reward, self._device) - self._baseline) * self._dt + \
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
        except IndexError:
            # Do nothing if not enough elements in the buffer
            pass

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
