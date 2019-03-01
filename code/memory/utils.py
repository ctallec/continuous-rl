from typing import Optional
from memory.buffer import MemorySampler, PrioritizedMemorySampler

def setup_memory(memory_size: int, alpha: Optional[float],
                 beta: Optional[float], batch_size: int):
    """Setup memory buffer."""
    args = dict(batch_size=batch_size,
                size=memory_size)

    if beta is not None:
        assert alpha is not None
        return PrioritizedMemorySampler(
            alpha=alpha, beta=beta, **args)
    return MemorySampler(**args)
