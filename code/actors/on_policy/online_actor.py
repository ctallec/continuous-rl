from abc import abstractmethod
import torch
from torch import Tensor
from distributions import Distribution
from distributions import Normal
from distributions import Categorical
from distributions import Independent
from abstract import ParametricFunction, Loggable, Tensorable
from stateful import CompoundStateful

class OnlineActor(CompoundStateful, Loggable):
    """Abstraction for online actor."""
    def __init__(self, policy_function: ParametricFunction,
                 dt: float, c_entropy: float) -> None:
        CompoundStateful.__init__(self)
        self._policy_function = policy_function

        self._c_entropy = c_entropy

    def act_noisy(self, obs: Tensorable) -> Tensor:
        distr = self._distr_generator(self._policy_function(obs))
        return distr.sample()

    @abstractmethod
    def act(self, obs: Tensorable) -> Tensor:
        pass

    def log(self) -> None:
        pass

    def to(self, device) -> "OnlineActor":
        CompoundStateful.to(self, device)
        self._device = device
        return self

    def actions_distr(self, obs: Tensorable) -> Distribution:
        """Returns the distributions on actions corresponding to the given obs."""
        return self._distr_generator(self._policy_function(obs))

    def policy(self, obs: Tensorable) -> Tensor:
        return self._policy_function(obs)

    @abstractmethod
    def actions(self, obs: Tensorable) -> Tensor:
        pass

class OnlineActorContinuous(OnlineActor):

    def __init__(self, policy_function: ParametricFunction,
                 dt: float, c_entropy: float) -> None:
        OnlineActor.__init__(self, policy_function, dt, c_entropy)

        self._distr_generator = lambda t: Independent(Normal(*t), 1)

    def act(self, obs: Tensorable) -> Tensor:
        action, _ = self._policy_function(obs)
        if not torch.isfinite(action).all():
            raise ValueError()
        return action

    def actions(self, obs: Tensorable) -> Tensor:
        return self._policy_function(obs)[0]

class OnlineActorDiscrete(OnlineActor):
    def __init__(self, policy_function: ParametricFunction,
                 dt: float, c_entropy: float) -> None:
        OnlineActor.__init__(self, policy_function, dt, c_entropy)
        self._distr_generator = lambda logits: Categorical(logits=logits)

    def act(self, obs: Tensorable) -> Tensor:
        return torch.argmax(self._policy_function(obs), dim=-1)

    def actions(self, obs: Tensorable) -> Tensor:
        return torch.softmax(self._policy_function(obs), dim=-1)[:, 0]
