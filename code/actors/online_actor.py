from abc import abstractmethod
import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from abstract import ParametricFunction, Loggable, Tensorable
from stateful import CompoundStateful
from optimizer import setup_optimizer
from memory.memorytrajectory import BatchTraj

class OnlineActor(CompoundStateful, Loggable):
    def __init__(self, policy_function: ParametricFunction,
                 lr: float, opt_name: str, dt: float,
                 c_entropy: float, weight_decay: float) -> None:
        CompoundStateful.__init__(self)
        self._policy_function = policy_function

        self._optimizer = setup_optimizer(
            self._policy_function.parameters(), opt_name=opt_name,
            lr=lr, dt=dt, inverse_gradient_magnitude=1, weight_decay=weight_decay)
        self._c_entropy = c_entropy

    @abstractmethod
    def _optimize_from_distr(self, distr: Distribution, traj: BatchTraj,
                             critic_value: Tensor) -> None:
        pass

    def act_noisy(self, obs: Tensorable) -> Tensor:
        distr = self._distr_generator(self._policy_function(obs))
        return distr.sample()

    @abstractmethod
    def act(self, obs: Tensorable) -> Tensor:
        pass

    def optimize(self, traj: BatchTraj, critic_value: Tensor) -> Tensor:
        traj = traj.to(self._device)
        distr = self._distr_generator(self._policy_function(traj.obs))
        self._optimize_from_distr(distr, traj, critic_value)

    def log(self) -> None:
        pass

    def to(self, device) -> "OnlineActor":
        CompoundStateful.to(self, device)
        self._device = device
        return self

    def policy(self, obs: Tensorable) -> Tensor:
        return self._policy_function(obs)

    @abstractmethod
    def actions(self, obs: Tensorable) -> Tensor:
        pass

class OnlineActorContinuous(OnlineActor):
    def __init__(self, policy_function: ParametricFunction,
                 lr: float, opt_name: str, dt: float,
                 c_entropy: float, weight_decay: float) -> None:
        OnlineActor.__init__(self, policy_function, lr, opt_name, dt,
                             c_entropy, weight_decay)

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
                 lr: float, opt_name: str, dt: float,
                 c_entropy: float, weight_decay: float) -> None:
        OnlineActor.__init__(self, policy_function, lr, opt_name, dt,
                             c_entropy, weight_decay)
        self._distr_generator = lambda logits: Categorical(logits=logits)

    def act(self, obs: Tensorable) -> Tensor:
        return torch.argmax(self._policy_function(obs), dim=-1)

    def actions(self, obs: Tensorable) -> Tensor:
        return torch.softmax(self._policy_function(obs), dim=-1)[:, 0]
