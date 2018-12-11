from torch import Tensor
from gym import Space
from gym.spaces import Box
from models import ContinuousPolicyMLP, NormalizedMLP
from abstract import Actor, ParametricFunction, Noise, Arrayable
from stateful import CompoundStateful
from optimizer import setup_optimizer

class ApproximateActor(CompoundStateful, Actor):
    def __init__(self, policy_function: ParametricFunction,
                 noise: Noise, lr: float, opt_name: str, dt: float,
                 weight_decay: float) -> None:
        CompoundStateful.__init__(self)
        self._policy_function = policy_function
        self._optimizer = setup_optimizer(
            self._policy_function.parameters(), opt_name=opt_name,
            lr=lr, dt=dt, inverse_gradient_magnitude=1, weight_decay=weight_decay)
        self._noise = noise

    def to(self, device):
        self._policy_function = self._policy_function.to(device)
        self._noise = self._noise.to(device)
        return self

    def act_noisy(self, obs: Arrayable) -> Arrayable:
        action = self._noise.perturb_output(
            obs, function=self._policy_function)
        self._noise.step()
        return action

    def act(self, obs: Arrayable, future=False) -> Tensor:
        return self._policy_function(obs)

    def optimize(self, loss: Tensor):
        self._optimizer.zero_grad()
        loss.mean().backward()
        self._optimizer.step()

    def log(self):
        pass

    @staticmethod
    def configure(
            action_space: Space, observation_space: Space,
            hidden_size: int, nb_layers: int, normalize: bool, noise: Noise,
            lr: float, optimizer: str, dt: float, weight_decay: float,
            **kwargs
    ):
        assert isinstance(action_space, Box)
        assert isinstance(observation_space, Box)
        nb_actions = action_space.shape[-1]
        nb_state_feats = observation_space.shape[-1]

        net_dict = dict(hidden_size=hidden_size, nb_layers=nb_layers)
        policy_function = ContinuousPolicyMLP(
            nb_inputs=nb_state_feats, nb_outputs=nb_actions, **net_dict)
        if normalize:
            policy_function = NormalizedMLP(policy_function)
        return ApproximateActor(policy_function, noise, lr,
                                optimizer, dt, weight_decay)
