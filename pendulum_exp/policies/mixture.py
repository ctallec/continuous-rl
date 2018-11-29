import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as f
from abstract import ParametricFunction, Arrayable, Noise, StateDict
from policies.shared import SharedAdvantagePolicy
from config import ApproximateAdvantagePolicyConfig
from convert import arr_to_th, check_array, th_to_arr
from optimizer import setup_optimizer
from nn import gmm_loss
from mylog import log
from logging import info

class ContinuousAdvantageMixturePolicy(SharedAdvantagePolicy):
    def __init__(self, policy_config: ApproximateAdvantagePolicyConfig,
                 val_function: ParametricFunction, adv_function: ParametricFunction,
                 policy_function: ParametricFunction, policy_noise: Noise,
                 device) -> None:
        super().__init__(policy_config, val_function, device)
        self._adv_function = adv_function
        self._policy_function = policy_function.to(device)
        self._policy_noise = policy_noise

        self._optimizers = (
            setup_optimizer(self._adv_function.parameters(),
                            opt_name=policy_config.optimizer,
                            lr=policy_config.lr, dt=self._dt,
                            inverse_gradient_magnitude=1,
                            weight_decay=policy_config.weight_decay),
            setup_optimizer(self._val_function.parameters(),
                            opt_name=policy_config.optimizer,
                            lr=policy_config.lr, dt=self._dt,
                            inverse_gradient_magnitude=1,
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

    def to_mixture(self, tens: Tensor, v_scale, logpi_scale) -> Tensor:
        v, logpi = tens[..., :2], tens[..., 2:]
        v = v * arr_to_th(v_scale, self._device)
        logpi = f.log_softmax(logpi + arr_to_th(logpi_scale, self._device), dim=-1)
        return v, logpi

    def compute_mixture_advantages(self, obs: Arrayable, action: Arrayable):
        adv, logpi = self.to_mixture(
            self._adv_function(obs, action),
            [[1, self._dt]], [[np.log(self._dt), 1]]) # (b, 4)
        max_action = self._policy_function(obs)
        max_adv, max_logpi = self.to_mixture(
            self._adv_function(obs, max_action),
            [[1, self._dt]], [[np.log(self._dt), 1]]) # (b, 4)

        mus = torch.cat([
            adv - max_adv,
            adv - torch.stack([max_adv[..., 1], max_adv[..., 0]], dim=-1)],
            dim=-1) # (b, 4)
        sigmas = arr_to_th([[
            np.sqrt(2), np.sqrt(2) * self._dt,
            np.sqrt(1 + self._dt ** 2), np.sqrt(1 + self._dt ** 2)]], self._device) # (1, 4)
        logpis = torch.cat([
            logpi + max_logpi,
            logpi + torch.stack([max_logpi[..., 1], max_logpi[..., 0]], dim=-1)],
            dim=-1) # (b, 4)

        adv = (adv * logpi.exp()).sum(dim=-1)
        max_adv = (max_adv * max_logpi.exp()).sum(dim=-1)

        return mus.unsqueeze(-1), sigmas.unsqueeze(-1), logpis, adv, max_adv # (*, 4, 1)

    def compute_advantages(self, obs: Arrayable, action: Arrayable):
        _, _, _, adv, max_adv = self.compute_mixture_advantages(obs, action)
        return adv, max_adv

    def _value(self, obs: Arrayable):
        mean, logpi = self.to_mixture(self._val_function(obs),
                                      [[1 / (1 - self._gamma), 1]],
                                      [[np.log(self._dt), 1]])
        return (mean * logpi.exp()).sum(-1)

    def _advantage(self, obs: Arrayable, action: Arrayable):
        adv, logpi = self.to_mixture(self._adv_function(obs, action),
                                     [[1, self._dt]],
                                     [[np.log(self._dt), 1]]) # (b, 4)
        return (adv * logpi.exp()).sum(-1)

    def compute_values(self, obs: Arrayable, next_obs: Arrayable, done: Tensor):
        mean, logpi = self.to_mixture(
            self._val_function(obs),
            [[1 / (1 - self._gamma), 1]],
            [[np.log(self._dt), 1]])
        next_v, next_logpi = self.to_mixture(
            self._val_function(next_obs),
            [[1 / (1 - self._gamma), 1]],
            [[np.log(self._dt), 1]])
        ref_v, ref_logpi = self.to_mixture(
            self._val_function(self._sampler.reference_obs),
            [[1 / (1 - self._gamma), 1]], [[np.log(self._dt), 1]])
        next_v, next_logpi = next_v.detach(), next_logpi.detach()
        ref_v, ref_logpi = ref_v.detach(), ref_logpi.detach()
        ref_v = (ref_v * ref_logpi.exp()).sum(dim=-1)
        mean_v = ref_v.mean()

        sigma = arr_to_th([[
            1, np.sqrt(self._dt)
        ]], self._device)

        v = (mean * logpi.exp()).sum(dim=-1)
        next_v = (next_v * next_logpi.exp()).sum(dim=-1) * (1 - done) -\
            done * mean_v * self._gamma / (1 - self._gamma)

        return mean.unsqueeze(-1), sigma.unsqueeze(-1), logpi, v, next_v, mean_v

    def learn(self):
        for net in self.networks():
            net.train()

        if self._count % self._steps_btw_train == self._steps_btw_train - 1:
            try:
                self.reset_log()
                for _ in range(self._learn_per_step):
                    obs, action, next_obs, reward, done, weights, time_limit = \
                        self._sampler.sample()

                    # don't update when a time limit is reached
                    if time_limit is not None:
                        weights = weights * (1 - time_limit)
                    reward = arr_to_th(reward, self._device)
                    weights = arr_to_th(check_array(weights), self._device)
                    done = arr_to_th(check_array(done).astype('float'), self._device)

                    # compute values
                    mu_v, sigma_v, logpi_v, v, next_v, mean_v = self.compute_values(obs, next_obs, done)

                    # compute advantages
                    mu_adv, sigma_adv, logpi_adv, adv, max_adv = self.compute_mixture_advantages(obs, action)

                    # compute advantage loss
                    expected_adv = reward * self._dt + self._gamma ** self._dt * next_v - v.detach() -\
                        (1 - done) * self._gamma * mean_v * self._dt
                    expected_adv = expected_adv.detach()
                    adv_loss = gmm_loss(expected_adv.unsqueeze(-1), mu_adv, sigma_adv, logpi_adv, reduce=False) * weights
                    self._sampler.observe(np.sqrt(np.abs(th_to_arr(adv_loss))))
                    adv_loss += ((max_adv ** 2) * weights)

                    # compute value loss
                    expected_v = reward * self._dt + self._gamma ** self._dt * next_v -\
                        (1 - done) * self._gamma * mean_v * self._dt + max_adv - adv
                    expected_v = expected_v.detach()
                    value_loss = gmm_loss(expected_v.unsqueeze(-1), mu_v, sigma_v, logpi_v, reduce=False) * weights

                    self.optimize_value(adv_loss, value_loss)
                    self.optimize_policy(max_adv)

                self.log()
                self.log_stats()
            except IndexError as e:
                # If not enough data in the buffer, do nothing
                raise e
                pass

    def act(self, obs: Arrayable):
        with torch.no_grad():
            action = self._policy_noise.perturb_output(
                obs, function=self._policy_function)
            self._policy_noise.step()
        return action

    def optimize_value(self, *losses: Tensor):
        self._optimizers[0].zero_grad()
        losses[0].mean().backward(retain_graph=True)
        self._optimizers[0].step()
        self._optimizers[1].zero_grad()
        losses[1].mean().backward(retain_graph=True)
        self._optimizers[1].step()

        # logging
        self._cum_loss += losses[0].mean().item()
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
        return th_to_arr(self._value(obs))

    def observe_evaluation(self, eval_return: float):
        self._schedulers[0].step(eval_return)
        self._schedulers[1].step(eval_return)
        self._schedulers[2].step(eval_return)

    def advantage(self, obs: Arrayable, action: Arrayable):
        return th_to_arr(self._advantage(obs, action))

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
        V = self.value(self._stats_obs).squeeze()
        actions = self._policy_function(self._stats_obs).cpu().numpy()
        return V, actions
