""" Define shared elements between continuous and discrete. """
import numpy as np
from abstract import Policy, Arrayable, ParametricFunction
from config import PolicyConfig
from memory import MemorySampler
from torch import Tensor
from convert import arr_to_th, check_array

class SharedAdvantagePolicy(Policy):
    def __init__(self, policy_config: PolicyConfig,
                 val_function: ParametricFunction, device) -> None:
        self._train = True
        self.reset()

        # parameters
        self._gamma = policy_config.gamma
        self._dt = policy_config.dt
        self._learn_per_step = policy_config.learn_per_step
        self._steps_btw_train = policy_config.steps_btw_train
        self._sampler = MemorySampler(
            size=policy_config.memory_size,
            batch_size=policy_config.batch_size)
        self._count = 0
        self._learn_count = 0
        self._device = device
        self._val_function = val_function

    def reset(self):
        # internals
        self._obs = np.array([])
        self._action = np.array([])
        self._reward = np.array([])
        self._next_obs = np.array([])
        self._done = np.array([])

    def act(self, obs: Arrayable):
        raise NotImplementedError

    def step(self, obs: Arrayable):
        if self._train:
            self._obs = obs

        action = self.act(obs)
        if self._train:
            self._action = action

        return action

    def observe(self,
                next_obs: Arrayable,
                reward: Arrayable,
                done: Arrayable):
        if self._train:
            self._count += 1
            self._next_obs = next_obs
            self._reward = reward
            self._done = done
            self._sampler.push(
                self._obs, self._action, self._next_obs, self._reward, self._done)
            self.learn()

    def learn(self):
        if self._count % self._steps_btw_train == self._steps_btw_train - 1:
            try:
                for _ in range(self._learn_per_step):
                    obs, action, next_obs, reward, done = self._sampler.sample()
                    indep_obs, _, _, _, _ = self._sampler.sample()
                    done = arr_to_th(check_array(done).astype('float'), self._device)
                    reward = arr_to_th(reward, self._device)

                    v = self._val_function(obs).squeeze()
                    next_v = self.compute_next_value(next_obs, done)
                    indep_v = self._val_function(indep_obs).squeeze().detach()
                    adv, max_adv = self.compute_advantages(
                        obs, action)

                    expected_v = reward * self._dt + \
                        self._gamma ** self._dt * next_v
                    dv = (expected_v - v) / self._dt - indep_v.mean()
                    bell_residual = dv - adv + max_adv

                    adv_update_loss = (bell_residual ** 2).mean()
                    adv_norm_loss = (max_adv ** 2).mean()
                    bell_loss = adv_update_loss + adv_norm_loss

                    self.optimize_value(bell_loss)
                    self.optimize_policy(max_adv)

                self.log()
            except IndexError:
                # If not enough data in the buffer, do nothing
                pass

    def optimize_value(self, *losses: Tensor):
        raise NotImplementedError()

    def optimize_policy(self, max_adv: Tensor):
        raise NotImplementedError()

    def compute_advantages(self, obs: Arrayable, action: Arrayable) -> Tensor:
        """Computes adv, max_adv."""
        raise NotImplementedError()

    def compute_next_value(self, next_obs: Arrayable, done: Arrayable) -> Tensor:
        """Also detach next value."""
        done = arr_to_th(check_array(done).astype('float'), self._device)
        next_v = self._val_function(next_obs).squeeze().detach()
        if self._gamma == 1:
            assert (1 - done).byte().all(), "Gamma set to 1. with a potentially"\
                "episodic problem..."
            return next_v
        return (1 - done) * next_v
