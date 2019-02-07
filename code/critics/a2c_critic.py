from torch import Tensor
from critics.online_critic import OnlineCritic
from gym.spaces import Box
from gym import Space
from models import MLP

class A2CCritic(OnlineCritic):
    def optimize(self, v: Tensor, v_target: Tensor) -> Tensor:
        self._optimizer.zero_grad()
        loss = ((v - v_target.detach()) ** 2).mean()
        loss.backward()
        self._optimizer.step()
        return loss

    @staticmethod
    def configure(dt: float, gamma: float, lr: float, optimizer: str,
                  observation_space: Space,
                  nb_layers: int, hidden_size: int,
                  noscale: bool) -> "OnlineCritic":

        assert isinstance(observation_space, Box)
        nb_state_feats = observation_space.shape[-1]
        v_function = MLP(nb_inputs=nb_state_feats, nb_outputs=1,
                         nb_layers=nb_layers, hidden_size=hidden_size)

        return A2CCritic(gamma, dt, lr, optimizer, v_function)
