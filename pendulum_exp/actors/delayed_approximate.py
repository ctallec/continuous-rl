from abstract import ParametricFunction, Noise, Arrayable
from actors.approximate import ApproximateActor
import copy
from torch import Tensor
from nn import soft_update
from gym import Space
from gym.spaces import Box
from models import ContinuousPolicyMLP, NormalizedMLP

class DelayedApproximateActor(ApproximateActor):
    def __init__(self, policy_function: ParametricFunction,
                 noise: Noise, lr: float, tau: float, opt_name: str, dt: float,
                 weight_decay: float) -> None:
        super().__init__(policy_function, noise, lr, opt_name, dt, weight_decay)
        self._target_policy_function = copy.deepcopy(self._policy_function)
        self._tau = tau

    def to(self, device):
        super().to(device)
        self._target_policy_function = self._target_policy_function.to(device)
        return self

    def act(self, obs: Arrayable, target=False) -> Tensor:
        if target:
            return self._target_policy_function(obs)
        return self._policy_function(obs)

    def optimize(self, loss: Tensor):
        super().optimize(loss)
        soft_update(self._policy_function, self._target_policy_function, self._tau)

    @staticmethod
    def configure(
            action_space: Space, observation_space: Space,
            hidden_size: int, nb_layers: int, normalize: bool, noise: Noise,
            lr: float, tau: float, optimizer: str, dt: float, weight_decay: float,
            **kwargs
    ):
        assert isinstance(action_space, Box)
        assert isinstance(observation_space, Box)
        nb_actions = action_space.shape[-1]
        nb_state_feats = observation_space.shape[-1]

        net_dict = dict(hidden_size=hidden_size, nb_layers=nb_layers)
        policy_function = ContinuousPolicyMLP(
            nb_inputs=nb_state_feats, nb_outputs=nb_actions, **net_dict)
        if normalize:
            policy_function = NormalizedMLP(policy_function)
        return DelayedApproximateActor(policy_function, noise, lr, tau,
                                       optimizer, dt, weight_decay)
