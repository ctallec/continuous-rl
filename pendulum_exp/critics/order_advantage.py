import copy
from abstract import Critic, Arrayable, ParametricFunction, Tensorable
import torch
from torch import Tensor
import numpy as np
from convert import arr_to_th, check_array, check_tensor
from optimizer import setup_optimizer
from gym.spaces import Box, Discrete
from models import MLP, ContinuousAdvantageMLP, NormalizedMLP
from stateful import CompoundStateful
from nn import soft_update

class OrderAdvantageCritic(CompoundStateful, Critic):
    def __init__(self,
                 dt: float, gamma: float, lr: float, tau: float, optimizer: str,
                 val_function: ParametricFunction, adv_function: ParametricFunction) -> None:
        CompoundStateful.__init__(self)
        self._reference_obs: Tensor = None
        self._val_function = val_function
        self._adv_function = adv_function
        self._target_val_function = copy.deepcopy(val_function)
        self._target_adv_function = copy.deepcopy(adv_function)

        self._adv_optimizer = \
            setup_optimizer(self._adv_function.parameters(),
                            opt_name=optimizer, lr=lr, dt=dt,
                            inverse_gradient_magnitude=1,
                            weight_decay=0)
        self._val_optimizer = \
            setup_optimizer(self._val_function.parameters(),
                            opt_name=optimizer, lr=lr, dt=dt,
                            inverse_gradient_magnitude=dt,
                            weight_decay=0)

        self._dt = dt
        self._gamma = gamma
        self._tau = tau

        self._device = 'cpu'

    def optimize(self, obs: Arrayable, action: Arrayable, max_action: Tensor,
                 next_obs: Arrayable, max_next_action: Tensor, reward: Arrayable,
                 done: Arrayable, time_limit: Arrayable, weights: Arrayable) -> Tensor:
        if self._reference_obs is None:
            self._reference_obs = arr_to_th(obs, self._device)

        obs = check_array(obs)
        next_obs = check_array(next_obs)
        action = arr_to_th(action, self._device).type_as(max_action)
        reward = arr_to_th(reward, self._device)
        done = arr_to_th(check_array(done).astype('float'), self._device)

        reference_v = self._val_function(self._reference_obs).squeeze()
        mean_v = reference_v.mean()

        v = self._val_function(obs).squeeze()
        bs = obs.shape[0]
        advs, sigmas = self.compute_advantages(
            np.concatenate([obs, obs], axis=0),
            torch.cat([action, max_action], dim=0))
        adv, max_adv = advs[:bs], advs[bs:2 * bs]
        sigma, max_sigma = sigmas[:bs], sigmas[bs:2 * bs]
        max_next_adv = self.compute_advantages(next_obs, max_next_action, target=True)[0]
        q = v + adv

        next_v = (1 - done) * (
            self._target_val_function(next_obs).squeeze() - self._dt * mean_v) - \
            done * self._gamma * mean_v / max(1 - self._gamma, 1e-5)
        next_q = next_v + (1 - done) * max_next_adv

        expected_q = (reward * self._dt + self._gamma ** self._dt * next_q).detach()

        critic_loss = .5 * ((expected_q - q) / (self._dt ** sigma)) ** 2 + \
            .5 * (max_adv / (self._dt ** max_sigma)) ** 2 + (sigma + max_sigma) * np.log(self._dt) / 2

        self._val_optimizer.zero_grad()
        self._adv_optimizer.zero_grad()
        critic_loss.mean().backward(retain_graph=True)
        self._val_optimizer.step()
        self._adv_optimizer.step()

        soft_update(self._val_function, self._target_val_function, self._tau)
        soft_update(self._adv_function, self._target_adv_function, self._tau)

        return critic_loss

    def compute_advantages(self, obs: Arrayable, action: Tensorable, target: bool = False) -> Tensor:
        func = self._adv_function if not target else self._target_adv_function
        if len(func.input_shape()) == 2:
            adv_full = func(obs, action).squeeze()
        else:
            adv_all = func(obs)
            action = check_tensor(action, self._device).long()
            adv_full = adv_all.gather(1, action.view(-1, 1)).squeeze()
        advs = adv_full[..., :2]
        sigma = torch.sigmoid(adv_full[..., 2] - 1)
        adv = (advs * torch.stack([sigma, 1 - sigma], dim=-1)).sum(dim=-1) * \
            (self._dt ** sigma)
        return adv, sigma

    def critic(self, obs: Arrayable, action: Tensorable, target: bool = False) -> Tensor:
        return self.compute_advantages(obs, action, target)[0]

    def log(self):
        pass

    def critic_function(self, target: bool = False):
        func = self._adv_function if not target else self._target_adv_function
        return func

    def to(self, device):
        CompoundStateful.to(self, device)
        self._device = device
        return self

    @staticmethod
    def configure(**kwargs):
        observation_space = kwargs['observation_space']
        action_space = kwargs['action_space']
        assert isinstance(observation_space, Box)
        nb_state_feats = observation_space.shape[-1]
        net_dict = dict(nb_layers=kwargs['nb_layers'], hidden_size=kwargs['hidden_size'])
        val_function = MLP(nb_inputs=nb_state_feats, nb_outputs=1, **net_dict)
        if isinstance(action_space, Discrete):
            nb_actions = action_space.n
            adv_function = MLP(nb_inputs=nb_state_feats, nb_outputs=nb_actions,
                               **net_dict)
        elif isinstance(action_space, Box):
            nb_actions = action_space.shape[-1]
            adv_function = ContinuousAdvantageMLP(
                nb_outputs=3, nb_state_feats=nb_state_feats, nb_actions=nb_actions,
                **net_dict)
        if kwargs['normalize']:
            val_function = NormalizedMLP(val_function)
            adv_function = NormalizedMLP(adv_function)
        return OrderAdvantageCritic(kwargs['dt'], kwargs['gamma'], kwargs['lr'], kwargs['tau'],
                                    kwargs['optimizer'], val_function, adv_function)
