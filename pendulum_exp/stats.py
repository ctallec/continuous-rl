""" Stat utilities. """
import numpy as np

class FloatingAvg:
    """ Computes floating average of a value. """
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._mean = None

    def step(self, value):
        """ Updates value of the mean and perform computations. """
        if self._mean is None:
            self._mean = value.copy()
        else:
            self._mean = (1 - self._alpha) * self._mean + self._alpha * value

    @property
    def mean(self) -> np.ndarray:
        """ Returns mean. """
        return self._mean

def epsilon_noise(arr: np.ndarray, epsilon: float) -> np.ndarray:
    """ Apply an epsilon greedy noise on arr. """
    mask = (np.random.uniform(0, 1, (arr.shape[0],)) < epsilon)
    return arr * (1 - mask) + (4 * np.random.randint(0, 2, (arr.shape[0],)) - 2) * mask
