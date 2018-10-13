""" Stat utilities. """
from typing import Optional, Callable
import numpy as np
import torch

class FloatingAvg:
    """ Computes floating average of a value. """
    def __init__(self, alpha: float,
                 decay: Optional[Callable[[int], float]] = None) -> None:
        self._count = 0
        self._alpha = alpha
        self._mean = 0
        self._decay = decay

    def step(self, value):
        """ Updates value of the mean and perform computations. """
        if self._mean is None:
            self._mean = value.mean()
        else:
            if self._decay is not None:
                alpha = self._alpha * self._decay(self._count)
                self._count += 1
            else:
                alpha = self._alpha
            self._mean = (1 - alpha) * self._mean + alpha * value.mean()

    @property
    def mean(self) -> np.ndarray:
        """ Returns mean. """
        if self._mean is None:
            return 0
        return self._mean

def epsilon_noise(arr: np.ndarray, epsilon: float) -> np.ndarray:
    """ Apply an epsilon greedy noise on arr. """
    mask = (np.random.uniform(0, 1, (arr.shape[0],)) < epsilon)
    return arr * (1 - mask) + (4 * np.random.randint(0, 2, (arr.shape[0],)) - 2) * mask

def penalize_mean(arr: torch.Tensor):
    """ Introduce a mean penalization. """
    mask = (torch.zeros_like(arr).uniform_() >= .5)
    length = min(torch.sum(mask), torch.sum(1 - mask)).item()
    return (arr[mask][:length] * arr[1 - mask][:length]).mean()
