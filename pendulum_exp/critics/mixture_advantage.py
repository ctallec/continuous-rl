from abstract import ParametricFunction, Arrayable, Tensorable
from optimizer import setup_optimizer
from torch import Tensor
from convert import arr_to_th, check_array, check_tensor
from critics.advantage import AdvantageCritic
from nn import gmm_loss
from gym import Space
from gym.spaces import Box, Discrete
from models import MLP, ContinuousAdvantageMLP, NormalizedMLP, MixtureNetwork
import torch
import torch.nn.functional as f
import numpy as np


class MixtureAdvantageCritic(AdvantageCritic):
    def __init__(self,
                 dt: float, gamma: float, lr: float, optimizer: str,
                 val_function: ParametricFunction, adv_function: ParametricFunction,
                 mixture_function: ParametricFunction) -> None:
        super().__init__(dt, gamma, lr, optimizer, val_function, adv_function)

        self._mixture_function = mixture_function

        self._val_optimizer = \
            setup_optimizer(self._val_function.parameters(),
                            opt_name=optimizer, lr=lr, dt=dt,
                            inverse_gradient_magnitude=1,
                            weight_decay=0)
        self._mixture_optimizer = \
            setup_optimizer(self._mixture_function.parameters(),
                            opt_name=optimizer, lr=lr, dt=dt,
                            inverse_gradient_magnitude=1,
                            weight_decay=0)

    def scale(self, val: Tensor, logpi: Tensor, val_scale, logpi_scale) -> Tensor:
        val = val * arr_to_th(val_scale, self._device)
        logpi = f.log_softmax(logpi + arr_to_th(logpi_scale, self._device), dim=-1)
        return val, logpi

    def to(self, device):
        super().to(device)
        self._mixture_function = self._mixture_function.to(device)
        return self

    def compute_mixture_advantage(self, obs: Arrayable, action: Arrayable):
        if len(self._adv_function.input_shape()) == 2:
            adv_mu, adv_logpi = self._adv_function(obs, action), self._mixture_function(obs, action)
        else:
            adv_all, logpi_all = self._adv_function(obs), self._mixture_function(obs)
            n_actions = adv_all.size(-1) // 2
            adv_all = adv_all.view(-1, 2, n_actions)
            logpi_all = adv_all.view(-1, 2, n_actions)
            action = check_tensor(action, self._device).long()
            adv_mu = adv_all.gather(2, action.view(-1, 1, 1).repeat(1, 2, 1)).squeeze()
            adv_logpi = logpi_all.gather(2, action.view(-1, 1, 1).repeat(1, 2, 1)).squeeze()

        adv_mu, adv_logpi = self.scale(
            adv_mu, adv_logpi,
            [[1, self._dt]], [[np.log(self._dt / (1 - self._dt)), 0]])

        adv = (adv_mu * adv_logpi.exp()).sum(dim=-1)

        return adv_mu, adv_logpi, adv

    def critic(self, obs: Arrayable, action: Tensorable, target: bool=False) -> Tensor:
        _, _, adv = self.compute_mixture_advantage(obs, action)
        return adv

    def critic_function(self, target: bool=False):
        return MixtureNetwork(self._adv_function, self._mixture_function,
                              [1, self._dt], [np.log(self._dt / (1 - self._dt)), 0])

    def optimize(self, obs: Arrayable, action: Arrayable, max_action: Tensor,
                 next_obs: Arrayable, max_next_action: Tensor, reward: Arrayable,
                 done: Arrayable, time_limit: Arrayable, weights: Arrayable) -> Tensor:
        if self._reference_obs is None:
            self._reference_obs = arr_to_th(obs, self._device)

        action = arr_to_th(action, self._device).type_as(max_action)
        reward = arr_to_th(reward, self._device)
        weights = arr_to_th(check_array(weights), self._device)
        done = arr_to_th(check_array(done).astype('float'), self._device)

        v = self._val_function(obs).squeeze()
        reference_v = self._val_function(self._reference_obs).squeeze()
        mean_v = reference_v.mean()
        next_v = (1 - done) * (
            self._val_function(next_obs).squeeze() - self._dt * mean_v) - \
            done * self._gamma * mean_v / max(1 - self._gamma, 1e-5)
        expected_adv = (reward * self._dt + self._gamma ** self._dt * next_v - v).detach()
        expected_adv = expected_adv.unsqueeze(-1)

        obs = check_array(obs)
        batch_size = obs.shape[0]
        adv_mus, adv_logpis, advs = self.compute_mixture_advantage(
            np.concatenate([obs, obs], axis=0),
            torch.cat([action, max_action], dim=0))
        adv_mu, max_adv_mu = adv_mus[:batch_size], adv_mus[batch_size:]
        adv_logpi, max_adv_logpi = adv_logpis[:batch_size], adv_logpis[batch_size:]
        adv, max_adv = advs[:batch_size], advs[batch_size:]

        gmm_mu = torch.cat(
            [
                adv_mu - max_adv_mu,
                adv_mu - torch.stack([max_adv_mu[..., 1], max_adv_mu[..., 0]], dim=-1)
            ], dim=-1).unsqueeze(-1)

        gmm_logpi = torch.cat(
            [
                adv_logpi + max_adv_logpi,
                adv_logpi + torch.stack([max_adv_logpi[..., 1], max_adv_logpi[..., 0]], dim=-1)
            ], dim=-1)

        sigma_coef = 1 / np.sqrt(self._dt)
        gmm_sigma = sigma_coef * torch.Tensor(
            [[np.sqrt(2), np.sqrt(2 * self._dt), np.sqrt(1 + self._dt), np.sqrt(1 + self._dt)]]
        ).to(self._device).unsqueeze(-1)

        adv_loss = gmm_loss(expected_adv, gmm_mu, gmm_sigma, gmm_logpi) + (max_adv ** 2)
        # remove minimal constant from adv_loss
        adv_loss = adv_loss - np.log(2 * np.pi) / 2 - np.log(2) / 2
        self._adv_optimizer.zero_grad()
        self._mixture_optimizer.zero_grad()
        adv_loss.mean().backward(retain_graph=True)
        self._adv_optimizer.step()
        self._mixture_optimizer.step()

        expected_v = (reward * self._dt + self._gamma ** self._dt * next_v - adv + max_adv).detach()
        value_loss = (v - expected_v) ** 2

        self._val_optimizer.zero_grad()
        value_loss.mean().backward(retain_graph=True)
        self._val_optimizer.step()

        return adv_loss

    @staticmethod
    def configure(dt: float, gamma: float, lr: float, optimizer: str,
                  action_space: Space, observation_space: Space,
                  nb_layers: int, hidden_size: int, normalize: bool, **kwargs):
        assert isinstance(observation_space, Box)
        nb_state_feats = observation_space.shape[-1]
        net_dict = dict(nb_layers=nb_layers, hidden_size=hidden_size)
        val_function = MLP(nb_inputs=nb_state_feats, nb_outputs=1, **net_dict)
        if isinstance(action_space, Discrete):
            nb_actions = action_space.n
            adv_function = MLP(nb_inputs=nb_state_feats, nb_outputs=2 * nb_actions,
                               **net_dict)
            mixture_function = MLP(nb_inputs=nb_state_feats, nb_outputs=2 * nb_actions,
                                   **net_dict)
        elif isinstance(action_space, Box):
            nb_actions = action_space.shape[-1]
            adv_function = ContinuousAdvantageMLP(
                nb_outputs=2, nb_state_feats=nb_state_feats, nb_actions=nb_actions,
                **net_dict)
            mixture_function = ContinuousAdvantageMLP(
                nb_outputs=2, nb_state_feats=nb_state_feats, nb_actions=nb_actions,
                **net_dict)

        if normalize:
            val_function = NormalizedMLP(val_function)
            adv_function = NormalizedMLP(adv_function)
            mixture_function = NormalizedMLP(mixture_function)
        return MixtureAdvantageCritic(dt, gamma, lr, optimizer,
                                      val_function, adv_function, mixture_function)
