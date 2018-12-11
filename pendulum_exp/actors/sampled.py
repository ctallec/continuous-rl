from typing import Tuple
from abstract import Actor, ParametricFunction, Noise, Arrayable, StateDict
from convert import check_array
import numpy as np
from torch import Tensor
import torch
from gym import Space
from gym.spaces import Box

class SampledActor(Actor):
    def __init__(self, critic: ParametricFunction,
                 noise: Noise, nb_samples: int, action_shape: Tuple[int, ...]) -> None:
        self._critic = critic
        self._noise = noise
        self._nb_samples = nb_samples
        self._action_shape = action_shape

    def act_noisy(self, obs: Arrayable) -> Arrayable:
        obs = check_array(obs)
        batch_size = obs.shape[0]
        o_dim = obs.ndim

        proposed_actions = np.random.uniform(
            -1, 1, [self._nb_samples, batch_size, *self._action_shape])
        advantages = self._noise.perturb_output(
            np.tile(obs, [self._nb_samples] + o_dim * [1]),
            proposed_actions, function=self._critic).squeeze()
        self._noise.step()
        action_idx = np.argmax(advantages, axis=0)
        return proposed_actions[action_idx, np.arange(obs.shape[0])]

    def act(self, obs: Arrayable, future=False) -> Tensor:
        obs = check_array(obs)
        proposed_actions = np.random.uniform(
            -1, 1, [self._nb_samples, obs.shape[0], *self._action_shape])
        actions = torch.argmax(
            self._critic(
                np.tile(obs, [self._nb_samples] + obs.ndim * [1]),
                proposed_actions).squeeze(), dim=0)
        return actions.cpu().numpy()

    def optimize(self, loss: Tensor):
        pass

    def load_state_dict(self, state_dict: StateDict):
        pass

    def state_dict(self) -> StateDict:
        return {}

    def to(self, device):
        self._noise = self._noise.to(device())
        return self

    def log(self):
        pass

    @staticmethod
    def configure(action_space: Space, noise: Noise, critic_function: ParametricFunction,
                  nb_samples: int, **kwargs):
        assert isinstance(action_space, Box)
        action_shape = action_space.shape
        return SampledActor(critic_function, noise, nb_samples, action_shape)
