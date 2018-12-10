from abstract import Critic, Arrayable, ParametricFunction, Tensorable
import torch
from torch import Tensor
import numpy as np
from convert import arr_to_th, check_array, check_tensor
from optimizer import setup_optimizer
from gym import Space
from gym.spaces import Box, Discrete
from models import MLP, ContinuousAdvantageMLP, NormalizedMLP
from stateful import CompoundStateful

class AdvantageCritic(CompoundStateful, Critic):
    def __init__(self,
                 dt: float, gamma: float, lr: float, optimizer: str,
                 val_function: ParametricFunction, adv_function: ParametricFunction) -> None:
        CompoundStateful.__init__(self)
        self._reference_obs: Tensor = None
        self._val_function = val_function
        self._adv_function = adv_function

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

        self._device = 'cpu'

    def optimize(self, obs: Arrayable, action: Arrayable, max_action: Tensor,
                 next_obs: Arrayable, max_next_action: Tensor, reward: Arrayable,
                 done: Arrayable, time_limit: Arrayable, weights: Arrayable) -> Tensor:
        if self._reference_obs is None:
            self._reference_obs = arr_to_th(obs, self._device)

        action = arr_to_th(action, self._device)
        reward = arr_to_th(reward, self._device)
        weights = arr_to_th(check_array(weights), self._device)
        done = arr_to_th(check_array(done).astype('float'), self._device)

        v = self._val_function(obs).squeeze()
        reference_v = self._val_function(self._reference_obs).squeeze()
        mean_v = reference_v.mean()
        next_v = (1 - done) * (
            self._val_function(next_obs).squeeze() - self._dt * mean_v) - \
            done * self._gamma * mean_v / max(1 - self._gamma, 1e-5)

        obs = check_array(obs)
        batch_size = obs.shape[0]
        advs = self.critic(
            np.concatenate([obs, obs], axis=0),
            torch.cat([action, max_action], dim=0))
        adv, max_adv = advs[:batch_size], advs[batch_size:]

        expected_v = (reward * self._dt + self._gamma ** self._dt * next_v).detach()
        bell_residual = (expected_v - v) / self._dt - adv + max_adv

        critic_loss = (bell_residual ** 2) + (max_adv ** 2)

        self._val_optimizer.zero_grad()
        self._adv_optimizer.zero_grad()
        critic_loss.mean().backward(retain_graph=True)
        self._val_optimizer.step()
        self._adv_optimizer.step()

        return critic_loss

    def critic(self, obs: Arrayable, action: Tensorable) -> Tensor:
        if len(self._adv_function.input_shape()) == 2:
            adv = self._adv_function(obs, action).squeeze()
        else:
            adv_all = self._adv_function(obs)
            action = check_tensor(action, self._device).long()
            adv = adv_all.gather(1, action.view(-1, 1)).squeeze()
        return adv

    @property
    def critic_function(self):
        return self._adv_function

    def to(self, device):
        self._val_function = self._val_function.to(device)
        self._adv_function = self._adv_function.to(device)
        self._device = device
        return self

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
            adv_function = MLP(nb_inputs=nb_state_feats, nb_outputs=nb_actions,
                               **net_dict)
        elif isinstance(action_space, Box):
            nb_actions = action_space.shape[-1]
            adv_function = ContinuousAdvantageMLP(
                nb_outputs=1, nb_state_feats=nb_state_feats, nb_actions=nb_actions,
                **net_dict)
        if normalize:
            val_function = NormalizedMLP(val_function)
            adv_function = NormalizedMLP(adv_function)
        return AdvantageCritic(dt, gamma, lr, optimizer, val_function, adv_function)
