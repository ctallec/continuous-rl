from typing import Optional
from abc import abstractmethod
from torch import Tensor

from abstract import Loggable, Arrayable, Tensorable
from cudaable import Cudaable
from actors.actor import Actor
from stateful import Stateful

class OfflineCritic(Stateful, Cudaable, Loggable):
    """Critic abstraction."""
    @abstractmethod
    def optimize(self, obs: Arrayable, action: Arrayable, max_action: Tensor,
                 next_obs: Arrayable, max_next_action: Tensor, reward: Arrayable,
                 done: Arrayable, time_limit: Arrayable, weights: Arrayable) -> Tensor:
        """Offline optimization. Typically uses variants of Q learning."""
        raise NotImplementedError()

    @abstractmethod
    def critic(self, obs: Arrayable, action: Tensorable, target: bool = False) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def value(self, obs: Arrayable, actor: Optional[Actor] = None) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def advantage(self, obs: Arrayable, action: Tensorable, actor: Actor) -> Tensor:
        pass
