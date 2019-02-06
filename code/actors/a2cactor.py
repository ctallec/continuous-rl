
from abc import abstractmethod
import copy
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
from gym.spaces import Box, Discrete
from models import ContinuousRandomPolicy, DiscreteRandomPolicy
from abstract import ParametricFunction, Loggable, Tensorable
from actors.actor import Actor
from stateful import CompoundStateful
from optimizer import setup_optimizer
from nn import soft_update
import random
from convert import check_tensor
from logging import info
from memory.memorytrajectory import BatchTraj


class A2CActor(CompoundStateful, Loggable):
    def __init__(self, policy_function: ParametricFunction,
                 lr: float, tau: float, opt_name: str, dt: float,
                 c_entropy:float, weight_decay: float) -> None:
        CompoundStateful.__init__(self)
        self._policy_function = policy_function

        self._optimizer = setup_optimizer(
            self._policy_function.parameters(), opt_name=opt_name,
            lr=lr, dt=dt, inverse_gradient_magnitude=1, weight_decay=weight_decay)
        self._tau = tau
        self._c_entropy = c_entropy


    def act_noisy(self, obs: Tensorable) -> Tensor:
        return self.act(obs)

    @abstractmethod
    def act(self, obs: Tensorable) -> Tensor:
        pass
        
    @abstractmethod
    def optimize(self, traj: BatchTraj, critic_value: Tensor) -> Tensor:
        pass

    def log(self) -> None:
        pass

    def to(self, device) -> "A2CActor":
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

        net_dict = dict(hidden_size=kwargs['hidden_size'], nb_layers=kwargs['nb_layers'])
        nb_state_feats = observation_space.shape[-1]
        if isinstance(action_space, Box):
            nb_actions = action_space.shape[-1]
            policy_generator = ContinuousRandomPolicy
            policy_function = ContinuousRandomPolicy(nb_state_feats, nb_actions,
                 **net_dict)
            actor_generator = A2CActorContinuous
        elif isinstance(action_space, Discrete):
            nb_actions = action_space.n
            policy_generator = DiscreteRandomPolicy
            actor_generator = A2CActorDiscrete
        policy_function = policy_generator(nb_state_feats, nb_actions, kwargs['nb_layers'], kwargs['hidden_size'])

        return actor_generator(policy_function, kwargs['lr'], kwargs['tau'],
                               kwargs['optimizer'], kwargs['dt'], kwargs['c_entropy'], 
                               kwargs['weight_decay'])



class A2CActorContinuous(A2CActor):
    def __init__(self, policy_function: ParametricFunction,
                 lr: float, tau: float, opt_name: str, dt: float,
                 c_entropy:float, weight_decay: float) -> None:
        A2CActor.__init__(self, policy_function, lr, tau, opt_name, dt,
                          c_entropy, weight_decay)
        

    def act(self, obs: Tensorable) -> Tensor:
        mu, sigma = self._policy_function(obs)
        eps = torch.randn_like(mu)
        action = mu + eps * sigma
        if not torch.isfinite(action).all():
            raise ValueError()
        return action


    def optimize(self, traj: BatchTraj,  critic_value: Tensor) -> None:
        traj = traj.to(self._device)
        action = traj.actions
        # batchobs = traj.obs.reshape((traj.batch_size * traj.length, *traj.obs.shape[2:]))     
        mu, sigma = self._policy_function(traj.obs)
        # mu, sigma = mu.reshape(action.shape), sigma.reshape(action.shape)

        logp_action = (- torch.log(sigma) -.5 * ((action - mu)/sigma) ** 2).sum(dim=-1)
        entropy = torch.log(sigma).sum(dim=-1).mean()
        # distr = torch.distributions.multivariate_normal.MultivariateNormal(
        #     mu, covariance_matrix=torch.diag(sigma ** 2))
        loss_critic = (- logp_action * critic_value.detach()).mean()
        loss = loss_critic - self._c_entropy * entropy

        info(f"loss_critic:{loss_critic.item():.0e}\tentropy:{entropy.item():.0e}")
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def actions(self, obs: Tensorable) -> Tensor:
        return self._policy_function(obs)[0]



class A2CActorDiscrete(A2CActor):
    def __init__(self, policy_function: ParametricFunction,
                 lr: float, tau: float, opt_name: str, dt: float,
                 c_entropy: float, weight_decay: float) -> None:
        A2CActor.__init__(self, policy_function, lr, tau, opt_name, dt,
                          c_entropy, weight_decay)

    def act_noisy(self, obs: Tensorable) -> Tensor:
        logp_actions = self._policy_function(obs)

        distr = torch.distributions.categorical.Categorical(
            logits=logp_actions)
        
        return distr.sample()

    def act(self, obs: Tensorable) -> Tensor:
        return torch.argmax(self._policy_function(obs), dim=-1)



    def optimize(self, traj: BatchTraj, critic_value: Tensor):
        traj = traj.to(self._device)
        actions = check_tensor(traj.actions, self._device)

        logits = self._policy_function(traj.obs)

        distr = Categorical(logits=logits)
        logp_actions = distr.log_prob(actions)
        entropy = distr.entropy()

        print(f"entropy:{entropy.mean().item()}")
        loss = - logp_actions * critic_value.detach() - self._c_entropy * entropy

        self._optimizer.zero_grad()
        loss.mean().backward()
        self._optimizer.step()

    def actions(self, obs: Tensorable) -> Tensor:
        return torch.argmax(self._policy_function(obs), dim=-1)
    


