import copy
from torch import Tensor
from gym.spaces import Box
from models import ContinuousPolicyMLP, NormalizedMLP
from abstract import Actor, ParametricFunction, Noise, Arrayable
from stateful import CompoundStateful
from optimizer import setup_optimizer
from nn import soft_update

class ApproximateActor(CompoundStateful, Actor):
    def __init__(self, policy_function: ParametricFunction,
                 noise: Noise, lr: float, tau: float, opt_name: str, dt: float,
                 weight_decay: float) -> None:
        CompoundStateful.__init__(self)
        self._policy_function = policy_function
        self._target_policy_function = copy.deepcopy(self._policy_function)

        self._optimizer = setup_optimizer(
            self._policy_function.parameters(), opt_name=opt_name,
            lr=lr, dt=dt, inverse_gradient_magnitude=1, weight_decay=weight_decay)
        self._noise = noise
        self._tau = tau

    def to(self, device):
        CompoundStateful.to(self, device)
        self._noise = self._noise.to(device)
        return self

    def act_noisy(self, obs: Arrayable) -> Arrayable:
        action = self._noise.perturb_output(
            obs, function=self._policy_function)
        self._noise.step()
        return action

    def act(self, obs: Arrayable, target=False) -> Tensor:
        if target:
            return self._target_policy_function(obs)
        return self._policy_function(obs)

    def optimize(self, loss: Tensor):
        self._optimizer.zero_grad()
        loss.mean().backward()
        self._optimizer.step()
        soft_update(self._policy_function, self._target_policy_function, self._tau)

    def log(self):
        pass

    @staticmethod
    def configure(**kwargs):
        action_space = kwargs['action_space']
        observation_space = kwargs['observation_space']
        assert isinstance(action_space, Box)
        assert isinstance(observation_space, Box)
        nb_actions = action_space.shape[-1]
        nb_state_feats = observation_space.shape[-1]

        net_dict = dict(hidden_size=kwargs['hidden_size'], nb_layers=kwargs['nb_layers'])
        policy_function = ContinuousPolicyMLP(
            nb_inputs=nb_state_feats, nb_outputs=nb_actions, **net_dict)
        if kwargs['normalize']:
            policy_function = NormalizedMLP(policy_function)
        return ApproximateActor(policy_function, kwargs['noise'], kwargs['lr'], kwargs['tau'],
                                kwargs['optimizer'], kwargs['dt'], kwargs['weight_decay'])
