import torch
from torch import Tensor
from typing import Optional, Tuple
from gym import Space
from gym.spaces import Box

from abstract import ParametricFunction, Tensorable
from optimizer import setup_optimizer

from models import MLP
from stateful import CompoundStateful
from memory.memorytrajectory import BatchTraj


class A2CCritic(CompoundStateful):
    def __init__(self, gamma: float, dt: float, lr: float, optimizer: str,
                 v_function: ParametricFunction) -> None:
        CompoundStateful.__init__(self)
        self._reference_obs: Tensor = None
        self._v_function = v_function
        self._optimizer = setup_optimizer(self._v_function.parameters(),
                                          opt_name=optimizer, lr=lr, dt=dt,
                                          inverse_gradient_magnitude=dt,
                                          weight_decay=0)
        self._gamma = gamma ** dt
        self._device = 'cpu'
        self._dt = dt

    def optimize(self, v: Tensor, v_target: Tensor) -> Tensor:
        self._optimizer.zero_grad()
        loss = ((v - v_target.detach()) ** 2).mean()
        loss.backward()
        self._optimizer.step()
        return loss

    def value(self, obs: Tensorable) -> Tensor:
        return self._v_function(obs)

    def value_batch(self, traj: BatchTraj, nstep: Optional[bool] = False) -> Tuple[Tensor, Tensor]:
        traj = traj.to(self._device)
        if not nstep:
            batchobs = traj.obs.reshape((traj.batch_size * traj.length, *traj.obs.shape[2:]))
            v = self._v_function(batchobs)
            v = v.reshape((traj.batch_size, traj.length))
            return v

        v = self.value_batch(traj, nstep=False)
        r = traj.rewards
        d = traj.done
        tl = traj.time_limit
        stop = torch.max(d, tl)
        btstrap = 1 - torch.max(1 - tl, d)
        btstrap[:, -1] = 1

        returns = (1 - btstrap) * r * self._dt + btstrap * v
        for t in reversed(range(0, traj.length - 1)):
            returns[:, t] += (1 - stop[:, t]) * (self._gamma * returns[:, t+1])

        return v, returns

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
                  noscale: bool) -> "A2CCritic":

        assert isinstance(observation_space, Box)
        nb_state_feats = observation_space.shape[-1]
        v_function = MLP(nb_inputs=nb_state_feats, nb_outputs=1,
                         nb_layers=nb_layers, hidden_size=hidden_size)

        return A2CCritic(gamma, dt, lr, optimizer, v_function)
