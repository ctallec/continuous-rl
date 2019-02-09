import numpy as np
from typing import Optional
import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.kl import kl_divergence
from abstract import ParametricFunction
from memory.memorytrajectory import BatchTraj
from logging import info
from actors.online_actor import OnlineActorContinuous, OnlineActorDiscrete
from gym.spaces import Box, Discrete
from models import ContinuousRandomPolicy, DiscreteRandomPolicy
from actors.online_actor import OnlineActor
from distributions import copy_distr



class PPOActor(OnlineActor):

    def reset_learning_step(self) -> None:
        self._old_distr: Optional[Distribution] = None
        self._old_logp: Optional[Tensor] = None

    def _optimize_from_distr(self, distr: Distribution, traj: BatchTraj, critic_value: Tensor) -> None:
        logp_action = distr.log_prob(traj.actions)

        if self._old_distr is None:
            self._old_distr = copy_distr(distr)
            self._old_logp = logp_action.clone().detach()
        logr = (logp_action - self._old_logp)

        r_clipped = torch.where(
            critic_value.detach() > 0,
            torch.clamp(logr, max=np.log(1+self._eps_clamp)),
            torch.clamp(logr, min=np.log(1-self._eps_clamp))
            ).exp()

        
        
        loss = - r_clipped * critic_value.detach()
        if self._c_entropy != 0.:
            loss -= self._c_entropy * distr.entropy()

        if self._c_kl != 0.:
            loss += self._c_kl * kl_divergence(self._old_distr, distr)

        loss = loss.mean()

        self._optimizer.zero_grad()
        loss.mean().backward()
        self._optimizer.step()

    

    @staticmethod
    def configure(**kwargs):
        action_space = kwargs['action_space']
        observation_space = kwargs['observation_space']
        assert isinstance(observation_space, Box)

        nb_state_feats = observation_space.shape[-1]
        if isinstance(action_space, Box):
            nb_actions = action_space.shape[-1]
            policy_generator, actor_generator = ContinuousRandomPolicy, PPOActorContinuous
        elif isinstance(action_space, Discrete):
            nb_actions = action_space.n
            policy_generator, actor_generator = DiscreteRandomPolicy, PPOActorDiscrete
        policy_function = policy_generator(nb_state_feats, nb_actions, kwargs['nb_layers'], kwargs['hidden_size'])

        return actor_generator(policy_function, kwargs['lr'], kwargs['optimizer'],
                               kwargs['dt'], kwargs['c_entropy'],
                               kwargs['weight_decay'], kwargs["eps_clamp"], kwargs["c_kl"])


class PPOActorContinuous(PPOActor, OnlineActorContinuous):
   def __init__(self, policy_function: ParametricFunction,
                 lr: float, opt_name: str, dt: float,
                 c_entropy: float, weight_decay: float, eps_clamp: float, c_kl: float):
        
        OnlineActorContinuous.__init__(self, policy_function=policy_function,
                 lr=lr, opt_name=opt_name, dt=dt,
                 c_entropy=c_entropy, weight_decay=weight_decay)
        self._eps_clamp = eps_clamp
        self._c_kl = c_kl
        self.reset_learning_step()

class PPOActorDiscrete(OnlineActorDiscrete, PPOActor):
    def __init__(self, policy_function: ParametricFunction,
                 lr: float, opt_name: str, dt: float,
                 c_entropy: float, weight_decay: float, eps_clamp: float, c_kl: float):
        
        OnlineActorDiscrete.__init__(self, policy_function=policy_function,
                 lr=lr, opt_name=opt_name, dt=dt,
                 c_entropy=c_entropy, weight_decay=weight_decay)
        self._eps_clamp = eps_clamp
        self._c_kl = c_kl
        self.reset_learning_step()

