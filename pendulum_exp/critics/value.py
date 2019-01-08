from abstract import Critic, Arrayable, ParametricFunction, Tensorable, Actor
from torch import Tensor
from typing import Optional
from convert import arr_to_th, check_array, check_tensor
from optimizer import setup_optimizer
from gym import Space
from gym.spaces import Box, Discrete
from models import MLP, ContinuousAdvantageMLP, NormalizedMLP
from stateful import CompoundStateful
import copy
from nn import soft_update

class ValueCritic(CompoundStateful, Critic):
    def __init__(self,
                 dt: float, gamma: float, lr: float, optimizer: str,
                 q_function: ParametricFunction, tau: float) -> None:
        CompoundStateful.__init__(self)
        self._reference_obs: Tensor = None
        self._q_function = q_function
        self._target_q_function = copy.deepcopy(self._q_function)
        self._tau = tau

        self._q_optimizer = \
            setup_optimizer(self._q_function.parameters(),
                            opt_name=optimizer, lr=lr, dt=dt,
                            inverse_gradient_magnitude=dt,
                            weight_decay=0)

        self._dt = dt
        self._gamma = gamma

        self._device = 'cpu'

    def optimize(self, obs: Arrayable, action: Arrayable, max_action: Tensor,
                 next_obs: Arrayable, max_next_action: Tensor, reward: Arrayable,
                 done: Arrayable, time_limit: Arrayable, weights: Arrayable) -> Tensor:
        action = arr_to_th(action, self._device)
        reward = arr_to_th(reward, self._device)
        weights = arr_to_th(check_array(weights), self._device)
        done = arr_to_th(check_array(done).astype('float'), self._device)

        obs = check_array(obs)
        next_obs = check_array(next_obs)
        q = self.critic(obs, action)
        q_next = self.critic(next_obs, max_next_action, target=True) * (1 - done)

        expected_q = (reward * self._dt + self._gamma ** self._dt * q_next).detach()
        critic_loss = (q - expected_q) ** 2

        self._q_optimizer.zero_grad()
        critic_loss.mean().backward(retain_graph=True)
        self._q_optimizer.step()

        soft_update(self._q_function, self._target_q_function, self._tau)

        return critic_loss

    def critic(self, obs: Arrayable, action: Tensorable, target: bool = False) -> Tensor:
        q_function = self._q_function if not target else self._target_q_function
        if len(q_function.input_shape()) == 2:
            q = q_function(obs, action).squeeze()
        else:
            q_all = q_function(obs)
            action = check_tensor(action, self._device).long()
            q = q_all.gather(1, action.view(-1, 1)).squeeze()
        return q

    def value(self, obs: Arrayable, actor: Optional[Actor] = None) -> Tensor:
        assert actor is not None
        return self.critic(obs, actor.act(obs))

    def advantage(self, obs: Arrayable, action: Tensorable, actor: Actor) -> Tensor:
        return self.critic(obs, action) - self.value(obs, actor)

    def log(self):
        pass

    def critic_function(self, target: bool = False):
        if target:
            return self._target_q_function
        return self._q_function

    def to(self, device):
        CompoundStateful.to(self, device)
        self._device = device
        return self

    @staticmethod
    def configure(dt: float, gamma: float, lr: float, optimizer: str,
                  action_space: Space, observation_space: Space,
                  nb_layers: int, hidden_size: int, normalize: bool,
                  tau: float, **kwargs):
        assert isinstance(observation_space, Box)
        nb_state_feats = observation_space.shape[-1]
        net_dict = dict(nb_layers=nb_layers, hidden_size=hidden_size)
        if isinstance(action_space, Discrete):
            nb_actions = action_space.n
            q_function = MLP(nb_inputs=nb_state_feats, nb_outputs=nb_actions,
                             **net_dict)
        elif isinstance(action_space, Box):
            nb_actions = action_space.shape[-1]
            q_function = ContinuousAdvantageMLP(
                nb_outputs=1, nb_state_feats=nb_state_feats, nb_actions=nb_actions,
                **net_dict)
        if normalize:
            q_function = NormalizedMLP(q_function)
        return ValueCritic(dt, gamma, lr, optimizer, q_function, tau)
