import torch
from torch import Tensor
from typing import Optional
from gym import Space
from gym.spaces import Box
import copy

from abstract import Arrayable, ParametricFunction, Tensorable
from convert import check_tensor
from optimizer import setup_optimizer

from models import MLP
from stateful import CompoundStateful
from nn import soft_update
from memory.memorytrajectory import BatchTraj


class A2CValue(CompoundStateful):
    def __init__(self, lr: float, dt: float, optimizer: str,
                 v_function: ParametricFunction, tau: float):
        CompoundStateful.__init__(self)
        self._lr = lr
        self._v_function = v_function
        self._target_v_function = copy.deepcopy(self._v_function)
        self._optimizer = setup_optimizer(self._v_function.parameters(),
                                          opt_name=optimizer, lr=lr, dt=dt,
                                          inverse_gradient_magnitude=self._dt,
                                          weight_decay=0)
        self._device = 'cpu'
        self._tau = tau

    def optimize(self, obs: Tensorable, expected_v: Tensorable) -> Tensor:
        expected_v = check_tensor(expected_v, self._device)
        v = self._v_function(obs)
        v_loss = ((v - expected_v) ** 2).mean()

        self._optimizer.zero_grad()
        v_loss.backward(retain_graph=True)
        self._optimizer.step()
        soft_update(self._v_function, self._target_v_function, self._tau)
        return v_loss

    def value(self, obs: Tensorable, target: Optional[bool]=False) -> Tensor:
        if target:
            v = self._target_v_function(obs)
        else:
            v = self._v_function(obs)
        v = v.squeeze(dim=1)
        if not torch.isfinite(v).all:
            raise ValueError()
        return v

    def log(self):
        pass

    def to(self, device) -> "A2CValue":
        CompoundStateful.to(self, device)
        self._device = device
        return self


class A2CCritic(CompoundStateful):
    def __init__(self, gamma: float, dt: float, lr: float, optimizer: str,
                 v_function: ParametricFunction, tau: float) -> None:
        CompoundStateful.__init__(self)
        self._reference_obs: Tensor = None
        self._a2cvalue = A2CValue(
            lr=lr, dt=dt, optimizer=optimizer, v_function=v_function, tau=tau)
        self._gamma = gamma ** dt
        self._device = 'cpu'
        self._dt

    def optimize(self, traj: BatchTraj) -> Tensor:
        v_nstep = self.value(traj, nstep=True, target=True).detach()
        obs = traj.obs[:,0]
        loss = self._a2cvalue.optimize(obs, v_nstep)
        print(f"loss_critic:{loss.item()}")
        return loss

    def critic(self, traj: BatchTraj, target: bool=False) -> Tensor:
        v_nstep = self.value(traj, nstep=True, target=True)
        v = self.value(traj, nstep=False, target=False)
        return v_nstep - v

    def value(self, traj: BatchTraj, nstep: Optional[bool]=False, target: Optional[bool]=False) -> Tensor:
        traj = traj.to(self._device)
        if nstep:
            lastobs, done = traj.obs[:, -1], traj.done[:, -1]

            discounts = self._gamma ** torch.arange(
                traj.length, dtype=traj.rewards.dtype, device=self._device)

            discounted_rewards = traj.rewards * discounts
            v = (discounted_rewards).sum(dim=1)
            v += (1 - done) * (self._gamma ** traj.length) * \
                self._a2cvalue.value(lastobs, target=target)
        else:
            v = self._a2cvalue.value(traj.obs[:, 0], target=target)
        return v

    def log(self) -> None:
        pass

    def to(self, device) -> "A2CCritic":
        CompoundStateful.to(self, device)
        self._device = device
        return self

    @staticmethod
    def configure(dt: float, gamma: float, lr: float, optimizer: str,
                  observation_space: Space,
                  nb_layers: int, hidden_size: int,
                  tau: float, noscale: bool) -> "A2CCritic":

        assert isinstance(observation_space, Box)
        nb_state_feats = observation_space.shape[-1]
        v_function = MLP(nb_inputs=nb_state_feats, nb_outputs=1,
                         nb_layers=nb_layers, hidden_size=hidden_size)

        return A2CCritic(gamma, dt, lr, optimizer, v_function, tau)
