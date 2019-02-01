from typing import List
import random

from abstract import Arrayable, ParametricFunction, Tensorable
from critics.critic import Critic
from actors.actor import Actor
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


# class Critic(Stateful, Cudaable, Loggable):
#     @abstractmethod
#     def optimize(self, obs: Arrayable, action: Arrayable, max_action: Tensor,
#                  next_obs: Arrayable, max_next_action: Tensor, reward: Arrayable,
#                  done: Arrayable, time_limit: Arrayable, weights: Arrayable) -> Tensor:
#         raise NotImplementedError()

#     @abstractmethod
#     def critic(self, obs: Arrayable, action: Tensorable, target: bool = False) -> Tensor:
#         raise NotImplementedError()

#     @abstractmethod
#     def value(self, obs: Arrayable, actor: Optional[Actor] = None) -> Tensor:
#         raise NotImplementedError()

#     @abstractmethod
#     def advantage(self, obs: Arrayable, action: Tensorable, actor: Actor) -> Tensor:
#         pass



class A2CValue(CompoundStateful):
    def __init__(self, lr: float, optimizer:str,
                 v_function: ParametricFunction, tau: float):
        CompoundStateful.__init__(self)
        self._lr = lr
        self._v_function = v_function
        self._target_v_function = copy.deepcopy(self._v_function)
        self._optimizer = setup_optimizer(self._v_function.parameters(),
                        opt_name=optimizer, lr=lr, dt=self._dt,
                        inverse_gradient_magnitude=self._dt,
                        weight_decay=0)
        self._device = 'cpu'

    def optimize(self, obs: Arrayable, expected_v:Tensorable) -> Tensor:
        obs = check_array(obs)
        v = self._v_function(obs)
        v_loss = (v - expected_v) ** 2

        self._optimizer.zero_grad()
        v_loss.mean().backward(retain_graph=True)
        self._optimizer.step()
        soft_update(self._v_function, self._target_v_function, self._tau)
        return v_loss

    def value(self, obs: Arrayable, target: Optional[bool] = False) -> Tensor:
        obs = check_array(obs)
        if target:
            v = self._target_v_function(obs)
        else:
            v = self._v_function(obs)
        return v

    def log(self):
        pass


    def to(self, device):
        CompoundStateful.to(self, device)
        self._device = device
        return self



class Trajectory:
    def __init__(self, 
                 obs: Optional[List[Arrayable]] = None, 
                 actions: Optional[List[Arrayable]] = None, 
                 rewards: Optional[List[float]] = None,
                 done: Optional[List[bool]] = None) -> None:
        
        if obs is None:
            obs = []
        if actions is None:
            actions = []
        if rewards is None:
            rewards = []
        if done is None:
            done = []
        self._obs = obs
        self._actions = actions
        self._rewards = rewards
        self._done = done

        l = len(self)
        assert len(self._rewards) == l and len(self._actions) == l and len(self._done) == l

    def append(self, obs: Arrayable, action: Arrayable, reward: float, done: bool) -> None:
        assert not self.isdone

        self._obs.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)
        self._done.append(done)

    def extract(self, length: int) -> "Trajectory":
        assert length <= len(self)
        start = random.randrange(len(self) - length + 1)
        stop = length - start
        obs = self._obs[start:stop]
        actions = self._actions[start:stop]
        rewards = self._rewards[start:stop]
        done = self._done[start:stop]

        return Trajectory(obs, actions, rewards, done)


    def __len__(self) -> int:
        return len(self._obs)

    @property
    def isdone(self) -> bool:
        if len(self) == 0:
            return False

        return self._done[-1]
    



class A2CValueCritic(CompoundStateful):
    def __init__(self, gamma: float, lr: float, optimizer: str,
                 v_function: ParametricFunction, tau: float) -> None:
        CompoundStateful.__init__(self)
        self._reference_obs: Tensor = None
        self._a2cvalue = A2CValue(lr=lr, optimizer=optimizer, v_function=v_function, tau=tau)

        self._device = 'cpu'


    def optimize(self, traj: Trajectory) -> Tensor:
        pass
        # action = arr_to_th(action, self._device)
        # reward = arr_to_th(reward, self._device)
        # weights = arr_to_th(check_array(weights), self._device)
        # done = arr_to_th(check_array(done).astype('float'), self._device)

        # obs = check_array(obs)
        # next_obs = check_array(next_obs)
        # q = self.critic(obs, action)
        # q_next = self.critic(next_obs, max_next_action, target=True) * (1 - done)

        # expected_q = (reward * self._dt + self._gamma ** self._dt * q_next).detach()
        # critic_loss = (q - expected_q) ** 2

        # self._q_optimizer.zero_grad()
        # critic_loss.mean().backward(retain_graph=True)
        # self._q_optimizer.step()

        # soft_update(self._q_function, self._target_q_function, self._tau)

        # return critic_loss

    def critic(self, obs: Arrayable, action: Tensorable, target: bool = False) -> Tensor:
        pass
        # q_function = self._q_function if not target else self._target_q_function
        # if len(q_function.input_shape()) == 2:
        #     q = q_function(obs, action).squeeze()
        # else:
        #     q_all = q_function(obs)
        #     action = check_tensor(action, self._device).long()
        #     q = q_all.gather(1, action.view(-1, 1)).squeeze()
        # return q

    def value(self, obs: Arrayable, actor: Optional[Actor] = None) -> Tensor:
        pass
        # assert actor is not None
        # return self.critic(obs, actor.act(obs))

    def advantage(self, obs: Arrayable, action: Tensorable, actor: Actor) -> Tensor:
        pass
        # return self.critic(obs, action) - self.value(obs, actor)

    def log(self):
        pass

    def critic_function(self, target: bool = False):
        pass
        # if target:
        #     return self._target_q_function
        # return self._q_function

    def to(self, device):
        pass
        # CompoundStateful.to(self, device)
        # self._device = device
        # return self

    @staticmethod
    def configure(dt: float, gamma: float, lr: float, optimizer: str,
                  action_space: Space, observation_space: Space,
                  nb_layers: int, hidden_size: int, normalize: bool,
                  tau: float, noscale: bool, **kwargs):
        pass
        # assert isinstance(observation_space, Box)
        # nb_state_feats = observation_space.shape[-1]
        # net_dict = dict(nb_layers=nb_layers, hidden_size=hidden_size)
        # if isinstance(action_space, Discrete):
        #     nb_actions = action_space.n
        #     q_function = MLP(nb_inputs=nb_state_feats, nb_outputs=nb_actions,
        #                      **net_dict)
        # elif isinstance(action_space, Box):
        #     nb_actions = action_space.shape[-1]
        #     q_function = ContinuousAdvantageMLP(
        #         nb_outputs=1, nb_state_feats=nb_state_feats, nb_actions=nb_actions,
        #         **net_dict)
        # if normalize:
        #     q_function = NormalizedMLP(q_function)
        # return ValueCritic(dt, gamma, lr, optimizer, q_function, tau, noscale)