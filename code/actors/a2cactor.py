
from abc import abstractmethod
import copy
import torch
from torch import Tensor
from gym.spaces import Box, Discrete
from models import ContinuousRandomPolicy, DiscreteRandomPolicy
from abstract import ParametricFunction, Arrayable, Loggable
from noises import Noise
from actors.actor import Actor
from stateful import CompoundStateful
from optimizer import setup_optimizer
from nn import soft_update
import random
from convert import check_array


class A2CActor(CompoundStateful, Loggable):
    def __init__(self, policy_function: ParametricFunction,
                 lr: float, tau: float, opt_name: str, dt: float,
                 c_entropy:float, weight_decay: float) -> None:
        CompoundStateful.__init__(self)
        self._policy_function = policy_function
        self._target_policy_function = copy.deepcopy(self._policy_function)

        self._optimizer = setup_optimizer(
            self._policy_function.parameters(), opt_name=opt_name,
            lr=lr, dt=dt, inverse_gradient_magnitude=1, weight_decay=weight_decay)
        self._tau = tau
        self._c_entropy = c_entropy


    def act_noisy(self, obs: Arrayable) -> Arrayable:
        return self.act(obs)

    @abstractmethod
    def act(self, obs: Arrayable, target=False) -> Tensor:
        pass
        
    @abstractmethod
    def optimize_critic(self, obs: Arrayable, action: Arrayable,  critic_value: Tensor):
        pass

    def log(self):
        pass

    def policy(self, obs: Arrayable, target: bool):
        if target:
            policy_fun = self._target_policy_function
        else:
            policy_fun = self._policy_function
        return policy_fun(obs)


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
        policy_function = policy_generator(nb_state_feats, nb_actions, kwargs['hidden_size'], kwargs['nb_layers'])

        return actor_generator(policy_function, kwargs['lr'], kwargs['tau'],
                               kwargs['optimizer'], kwargs['dt'], kwargs['c_entropy'], 
                               kwargs['weight_decay'])



class A2CActorContinuous(A2CActor):
    def __init__(self, policy_function: ParametricFunction,
                 lr: float, tau: float, opt_name: str, dt: float,
                 c_entropy:float, weight_decay: float) -> None:
        A2CActor.__init__(self, policy_function, lr, tau, opt_name, dt,
                          c_entropy, weight_decay)
        

    def act(self, obs: Arrayable, target=False) -> Tensor:
        mu, sigma = self.policy(obs, target)
        eps = torch.randn_like(mu)
        return mu + eps * sigma

    def optimize_critic(self, obs: Arrayable, action: Arrayable,  critic_value: Tensor):
        action = check_array(action)        
        mu, sigma = self._policy_function(obs)

        logp_action = - torch.log(sigma).sum(dim=1) -.5 * (action - mu)/sigma ** 2
        entropy = torch.log(sigma).sum(dim=1)

        loss = - logp_action * critic_value.detach() - self._c_entropy * entropy

        self._optimizer.zero_grad()
        loss.mean().backward()
        self._optimizer.step()
        soft_update(self._policy_function, self._target_policy_function, self._tau)



class A2CActorDiscrete(A2CActor):
    def __init__(self, policy_function: ParametricFunction,
                 lr: float, tau: float, opt_name: str, dt: float,
                 c_entropy: float, weight_decay: float) -> None:
        A2CActor.__init__(self, policy_function, lr, tau, opt_name, dt,
                          c_entropy, weight_decay)

    def act(self, obs: Arrayable, target=False) -> Tensor:
        p_actions = self.policy(obs, target).exp()
        return random.choices(list(range(self._nb_action)), weights=p_actions)

    def optimize_critic(self, obs: Arrayable, action: Arrayable, critic_value: Tensor):
        action = check_array(action)
        logp_action = self._policy_function(obs).gather(1, action.view(-1, 1)).squeeze()
        loss = - logp_action * critic_value.detach()

        self._optimizer.zero_grad()
        loss.mean().backward()
        self._optimizer.step()
        soft_update(self._policy_function, self._target_policy_function, self._tau)

    


