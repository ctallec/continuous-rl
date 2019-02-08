from abc import abstractmethod
from logging import info
import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete
from models import ContinuousRandomPolicy, DiscreteRandomPolicy
from abstract import ParametricFunction, Loggable, Tensorable
from stateful import CompoundStateful
from optimizer import setup_optimizer
from memory.memorytrajectory import BatchTraj
from distributions import DiagonalNormal

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

    # @abstractmethod
    def _optimize_from_distr(self, distr: Distribution, traj: BatchTraj,
                             critic_value: Tensor) -> None:
        action = traj.actions
        logp_action = distr.log_prob(action)
        entropy = distr.entropy()

        loss_critic = (- logp_action * critic_value.detach()).mean()
        loss = loss_critic - self._c_entropy * entropy

        info(f"loss_critic:{loss_critic.mean().item():3.2e}\t"
             f"entropy:{entropy.mean().item():3.2e}")
        self._optimizer.zero_grad()
        loss.mean().backward()
        self._optimizer.step()

    def act_noisy(self, obs: Tensorable) -> Tensor:
        distr = self._distr_generator(self._policy_function(obs))
        return distr.sample()

    @abstractmethod
    def act(self, obs: Tensorable) -> Tensor:
        pass

    # @abstractmethod
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

    @staticmethod
    def configure(**kwargs):
        action_space = kwargs['action_space']
        observation_space = kwargs['observation_space']
        assert isinstance(observation_space, Box)

        # net_dict = dict(hidden_size=kwargs['hidden_size'], nb_layers=kwargs['nb_layers'])
        nb_state_feats = observation_space.shape[-1]
        if isinstance(action_space, Box):
            nb_actions = action_space.shape[-1]
            policy_generator = ContinuousRandomPolicy
            # policy_function = ContinuousRandomPolicy(nb_state_feats, nb_actions,
            #                                          **net_dict)
            actor_generator = OnlineActorContinuous
        elif isinstance(action_space, Discrete):
            nb_actions = action_space.n
            policy_generator = DiscreteRandomPolicy
            actor_generator = OnlineActorDiscrete
        policy_function = policy_generator(nb_state_feats, nb_actions, kwargs['nb_layers'], kwargs['hidden_size'])

        return actor_generator(policy_function, kwargs['lr'], kwargs['optimizer'],
                               kwargs['dt'], kwargs['c_entropy'],
                               kwargs['weight_decay'])

class OnlineActorContinuous(OnlineActor):
    def __init__(self, policy_function: ParametricFunction,
                 lr: float, opt_name: str, dt: float,
                 c_entropy: float, weight_decay: float) -> None:
        OnlineActor.__init__(self, policy_function, lr, opt_name, dt,
                             c_entropy, weight_decay)

        self._distr_generator = lambda t: DiagonalNormal(*t)

    # def act_noisy(self, obs: Tensorable) -> Tensor:
    #     mu, sigma = self._policy_function(obs)

    #     return DiagonalNormal(mu, sigma).sample()

    def act(self, obs: Tensorable) -> Tensor:
        action, _ = self._policy_function(obs)
        if not torch.isfinite(action).all():
            raise ValueError()
        return action

    # def optimize(self, traj: BatchTraj, critic_value: Tensor) -> None:
    #     traj = traj.to(self._device)
    #     mu, sigma = self._policy_function(traj.obs)
    #     distr = DiagonalNormal(mu, sigma)
    #     self._optimize_from_distr(distr, traj, critic_value)

    def actions(self, obs: Tensorable) -> Tensor:
        return self._policy_function(obs)[0]

class OnlineActorDiscrete(OnlineActor):
    def __init__(self, policy_function: ParametricFunction,
                 lr: float, opt_name: str, dt: float,
                 c_entropy: float, weight_decay: float) -> None:
        OnlineActor.__init__(self, policy_function, lr, opt_name, dt,
                             c_entropy, weight_decay)
        self._distr_generator = lambda logits: Categorical(logits=logits)


    # def act_noisy(self, obs: Tensorable) -> Tensor:
    #     logp_actions = self._policy_function(obs)

    #     distr = torch.distributions.categorical.Categorical(
    #         logits=logp_actions)

    #     return distr.sample()

    def act(self, obs: Tensorable) -> Tensor:
        return torch.argmax(self._policy_function(obs), dim=-1)

    # def optimize(self, traj: BatchTraj, critic_value: Tensor):
    #     traj = traj.to(self._device)
    #     logits = self._policy_function(traj.obs)

    #     distr = Categorical(logits=logits)
    #     self._optimize_from_distr(distr, traj, critic_value)

    def actions(self, obs: Tensorable) -> Tensor:
        return torch.softmax(self._policy_function(obs), dim=-1)[:, 0]
