from torch import Tensor
from torch.distributions import Distribution
from actors.on_policy.online_actor import OnlineActorContinuous, OnlineActorDiscrete
from gym.spaces import Box, Discrete
from models import ContinuousRandomPolicy, DiscreteRandomPolicy, NormalizedMLP

def loss(distr: Distribution, actions: Tensor,
         critic_value: Tensor, c_entropy: float) -> Tensor:
    logp_action = distr.log_prob(actions)
    entropy = distr.entropy()

    loss_critic = (- logp_action * critic_value.detach()).mean()
    return loss_critic - c_entropy * entropy.mean()

class A2CActorContinuous(OnlineActorContinuous):
    def loss(self, distr: Distribution, actions: Tensor, critic_value: Tensor) -> Tensor:
        return loss(distr, actions, critic_value, self._c_entropy)

class A2CActorDiscrete(OnlineActorDiscrete):
    def loss(self, distr: Distribution, actions: Tensor, critic_value: Tensor) -> Tensor:
        return loss(distr, actions, critic_value, self._c_entropy)

# this is a dummy class
class A2CActor(object):
    @staticmethod
    def configure(**kwargs):
        action_space = kwargs['action_space']
        observation_space = kwargs['observation_space']
        assert isinstance(observation_space, Box)

        nb_state_feats = observation_space.shape[-1]
        if isinstance(action_space, Box):
            nb_actions = action_space.shape[-1]
            policy_generator, actor_generator = ContinuousRandomPolicy, A2CActorContinuous
        elif isinstance(action_space, Discrete):
            nb_actions = action_space.n
            policy_generator, actor_generator = DiscreteRandomPolicy, A2CActorDiscrete
        policy_function = policy_generator(nb_state_feats, nb_actions, kwargs['nb_layers'], kwargs['hidden_size'])

        if kwargs['normalize']:
            policy_function = NormalizedMLP(policy_function)

        return actor_generator(policy_function, kwargs['dt'], kwargs['c_entropy'])
