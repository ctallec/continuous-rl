from mylog import log
from logging import info
from memory.memorytrajectory import Trajectory
from policies.online_policy import OnlinePolicy


class A2CPolicy(OnlinePolicy):
    def learn(self) -> None:
        if (self._count + 1) % self._T != 0:
            return None
        traj = Trajectory.tobatch(*self._current_trajectories)

        v, v_target = self._critic.value_batch(traj)

        critic_loss = self._critic.optimize(v, v_target)
        critic_value = v_target - v
        self._actor.optimize(traj, critic_value)

        critic_loss = critic_loss.mean().item()
        critic_value = critic_value.mean().item()

        info(f'At step {self._count}, critic loss: {critic_loss}')
        info(f'At step {self._count}, critic value: {critic_value}')
        log("loss/critic", critic_loss, self._count)
        log("value/critic", critic_value, self._count)
        self._actor.log()
        self._critic.log()
