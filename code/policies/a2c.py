from typing import Union
from mylog import log
from logging import info
from memory.memorytrajectory import Trajectory
from policies.online_policy import OnlinePolicy
from optimizer import setup_optimizer
from itertools import chain
from critics.a2c_critic import A2CCritic
from actors.a2c_actor import A2CActorContinuous, A2CActorDiscrete

TypeA2CActor = Union[A2CActorContinuous, A2CActorDiscrete]

class A2CPolicy(OnlinePolicy):
    def __init__(self, T: int, nb_train_env: int,
                 actor: TypeA2CActor, critic: A2CCritic, opt_name: str, lr: float,
                 dt: float, weight_decay: float):
        OnlinePolicy.__init__(self, T=T, nb_train_env=nb_train_env, actor=actor,
                              critic=critic)

        self._optimizer = setup_optimizer(
            chain(self._actor._policy_function.parameters(), self._critic._v_function.parameters()),
            opt_name=opt_name,
            lr=lr, dt=dt, inverse_gradient_magnitude=1, weight_decay=weight_decay)

    def learn(self) -> None:
        if (self._count + 1) % self._T != 0:
            return None
        traj = Trajectory.tobatch(*self._current_trajectories)
        traj = traj.to(self._device)
        v, v_target = self._critic.value_batch(traj)

        critic_loss = self._critic.loss(v, v_target)
        critic_value = v_target - v

        obs = traj.obs
        actions = traj.actions
        distr = self._actor.actions_distr(obs)

        actor_loss = self._actor.loss(distr=distr, actions=actions,
                                      critic_value=critic_value)

        loss = critic_loss + actor_loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        critic_loss = critic_loss.mean().item()
        critic_value = critic_value.mean().item()
        actor_loss = actor_loss.mean().item()

        info(f'At step {self._count}, critic loss: {critic_loss}')
        info(f'At step {self._count}, critic value: {critic_value}')
        info(f'At step {self._count}, actor loss: {actor_loss}')
        log("loss/critic", critic_loss, self._count)
        log("value/critic", critic_value, self._count)
        log("loss/actor", actor_loss, self._count)
        self._actor.log()
        self._critic.log()
