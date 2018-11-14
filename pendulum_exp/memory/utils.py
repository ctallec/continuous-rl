from memory.buffer import MemorySampler, PrioritizedMemorySampler
from config import PolicyConfig

def setup_memory(policy_config: PolicyConfig):
    args = dict(batch_size=policy_config.batch_size,
                size=policy_config.memory_size)

    if policy_config.beta is not None:
        return PrioritizedMemorySampler(
            beta=policy_config.beta, **args)
    return MemorySampler(**args)
