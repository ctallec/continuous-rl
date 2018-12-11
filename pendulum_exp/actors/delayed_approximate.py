from abstract import ParametricFunction, Noise, Arrayable
from actors.approximate import ApproximateActor
import copy
from torch import Tensor
from nn import soft_update

class DelayedApproximateActor(ApproximateActor):
    def __init__(self, policy_function: ParametricFunction,
                 noise: Noise, lr: float, opt_name: str, dt: float,
                 weight_decay: float, tau: float) -> None:
        super().__init__(policy_function, noise, lr, opt_name, dt, weight_decay)
        self._target_policy_function = copy.deepcopy(self._policy_function)
        self._tau = tau

    def to(self, device):
        super().to(device)
        self._target_policy_function = self._target_policy_function.to(device)
        return self

    def act(self, obs: Arrayable, future=False) -> Tensor:
        if future:
            return self._target_policy_function(obs)
        return self._policy_function(obs)

    def optimize(self, loss: Tensor):
        super().optimize(loss)
        soft_update(self._policy_function, self._target_policy_function, self._tau)
