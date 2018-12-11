from actors.discrete import DiscreteActor
from abstract import ParametricFunction, Noise, Arrayable
from torch import Tensor

class DelayedDiscreteActor(DiscreteActor):
    def __init__(self,
                 critic: ParametricFunction,
                 target_critic: ParametricFunction,
                 noise: Noise) -> None:
        super().__init__(critic, noise)
        self._target_critic = target_critic

    def act(self, obs: Arrayable, target: bool=False) -> Tensor:
        critic = self._critic if not target else self._target_critic
        pre_action = critic(obs)
        return pre_action.argmax(dim=-1)

    @staticmethod
    def configure(
            critic_function: ParametricFunction, target_critic_function: ParametricFunction,
            noise: Noise, **kwargs
    ):
        return DelayedDiscreteActor(critic_function, target_critic_function, noise)
