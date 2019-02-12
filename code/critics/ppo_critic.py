import torch
from torch import Tensor
from critics.online_critic import OnlineCritic
from gym.spaces import Box
from gym import Space
from models import MLP
from abstract import ParametricFunction

class PPOCritic(OnlineCritic):
    def __init__(self, gamma: float, dt: float,
                 v_function: ParametricFunction, eps_clamp: float):
        OnlineCritic.__init__(self, gamma=gamma, dt=dt,
                              v_function=v_function)
        self._eps_clamp = eps_clamp

    def loss(self, v: Tensor, v_target: Tensor, old_v: Tensor) -> Tensor:
        assert old_v.shape == v.shape and v_target.shape == v.shape
        loss_unclipped = ((v - v_target.detach()) ** 2).mean()
        v_clipped = old_v + torch.clamp(v-old_v, -self._eps_clamp, self.eps_clamp)
        loss_clipped = ((v_clipped - v_target.detach()) ** 2).mean()
        return .5 * (loss_clipped + loss_unclipped)

    @staticmethod
    def configure(dt: float, gamma: float, observation_space: Space,
                  nb_layers: int, hidden_size: int,
                  noscale: bool, eps_clamp: float) -> "OnlineCritic":

        assert isinstance(observation_space, Box)
        nb_state_feats = observation_space.shape[-1]
        v_function = MLP(nb_inputs=nb_state_feats, nb_outputs=1,
                         nb_layers=nb_layers, hidden_size=hidden_size)

        return PPOCritic(gamma, dt, v_function, eps_clamp)
