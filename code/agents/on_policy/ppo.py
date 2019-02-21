from itertools import chain
from logging import info

import torch

from mylog import log
from memory.trajectory import Trajectory
from optimizer import setup_optimizer

from agents.on_policy.online_agent import OnlineAgent
from actors.on_policy.online_actor import OnlineActor
from critics.on_policy.online_critic import OnlineCritic


class PPOAgent(OnlineAgent):

    def __init__(self, T: int, nb_train_env: int, actor: OnlineActor,
                 critic: OnlineCritic, learn_per_step: int, batch_size: int,
                 opt_name: str, lr: float, dt: float, weight_decay: float):

        OnlineAgent.__init__(self, T=T, nb_train_env=nb_train_env, actor=actor,
                             critic=critic)
        self._learn_per_step = learn_per_step
        self._batch_size = batch_size

        self._optimizer = setup_optimizer(
            chain(self._actor._policy_function.parameters(), self._critic._v_function.parameters()),
            opt_name=opt_name,
            lr=lr, dt=dt, inverse_gradient_magnitude=1, weight_decay=weight_decay)

    def learn(self) -> None:
        if (self._count + 1) % self._T != 0:
            return None

        traj = Trajectory.tobatch(*self._current_trajectories).to(self._device)
        v, v_target = self._critic.value_batch(traj)

        obs_flat = traj.obs.flatten(0, 1)
        actions_flat = traj.actions.flatten(0, 1)
        distr_flat = self._actor._distr_generator(
            self._actor.policy(traj.obs.flatten(0, 1)))
        old_distr = self._actor.copy_distr(distr_flat)
        old_logp = distr_flat.log_prob(actions_flat).clone().detach()
        old_v = v.flatten().clone().detach()
        critic_value_flat = (v_target - v).flatten()
        full_batch_size = traj.length * traj.batch_size


        for ep in range(self._learn_per_step):
            perm = torch.randperm(full_batch_size)

            for start in range(0, full_batch_size // self._batch_size, self._batch_size):
                idxs = perm[start:start+self._batch_size]

                v = self._critic.value(obs_flat[idxs]).squeeze()
                critic_loss = self._critic.loss(
                    v, v_target.flatten()[idxs].detach(), old_v[idxs].detach())
                loss_actor = self._actor.loss(
                    distr=self._actor._distr_generator(self._actor.policy(obs_flat[idxs])),
                    actions=actions_flat[idxs], critic_value=critic_value_flat[idxs],
                    old_logp=old_logp[idxs],
                    old_distr=self._actor.distr_minibatch(old_distr, idxs)
                )

                loss = loss_actor + critic_loss
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()


        critic_loss = critic_loss.mean().item()
        critic_value = critic_value_flat.mean().item()
        info(f'At step {self._count}, critic loss: {critic_loss}')
        info(f'At step {self._count}, critic value: {critic_value}')
        log("loss/critic", critic_loss, self._count)
        log("value/critic", critic_value, self._count)
        self._actor.log()
        self._critic.log()
