"""Define critic abstraction."""
from typing import Union
from critics.off_policy.offline_critic import OfflineCritic
from critics.on_policy.online_critic import OnlineCritic

Critic = Union[OnlineCritic, OfflineCritic]
