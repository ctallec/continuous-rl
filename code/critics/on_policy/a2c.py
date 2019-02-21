from torch import Tensor
from critics.on_policy.online_critic import OnlineCritic
from gym.spaces import Box
from gym import Space
from models import MLP, NormalizedMLP

class A2CCritic(OnlineCritic):
    def loss(self, v: Tensor, v_target: Tensor) -> Tensor:
        return ((v - v_target.detach()) ** 2).mean()

    @staticmethod
    def configure(dt: float, gamma: float,
                  observation_space: Space,
                  nb_layers: int, hidden_size: int,
                  noscale: bool, normalize: bool) -> "OnlineCritic":

        assert isinstance(observation_space, Box)
        nb_state_feats = observation_space.shape[-1]
        v_function = MLP(nb_inputs=nb_state_feats, nb_outputs=1,
                         nb_layers=nb_layers, hidden_size=hidden_size)
        if normalize:
            v_function = NormalizedMLP(v_function)

        return A2CCritic(gamma, dt, v_function)
