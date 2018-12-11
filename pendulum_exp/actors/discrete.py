from torch import Tensor
from abstract import Actor, ParametricFunction, Noise, Arrayable, StateDict

class DiscreteActor(Actor):
    def __init__(self, critic: ParametricFunction, noise: Noise) -> None:
        self._critic = critic
        self._noise = noise

    def act_noisy(self, obs: Arrayable) -> Arrayable:
        pre_action = self._noise.perturb_output(
            obs, function=self._critic)
        self._noise.step()
        return pre_action.argmax(axis=-1)

    def act(self, obs: Arrayable, target=False) -> Tensor:
        pre_action = self._critic(obs)
        return pre_action.argmax(dim=-1)

    def optimize(self, loss: Tensor):
        pass

    def state_dict(self) -> StateDict:
        return {}

    def load_state_dict(self, state_dict):
        pass

    def to(self, device):
        self._noise = self._noise.to(device)
        return self

    def log(self):
        pass

    @staticmethod
    def configure(
            critic_function: ParametricFunction, noise: Noise, **kwargs
    ):
        return DiscreteActor(critic_function, noise)
