""" Define noise model. """
from typing import List, Tuple, Optional
import numpy as np

class OrnsteinUlhenbeck(object): # pylint: disable=too-few-public-methods
    """ Ornstein Ulhenbeck process. """
    def __init__(self, # pylint: disable=too-many-arguments
                 dt: float,
                 mu: np.ndarray,
                 sigma: np.ndarray,
                 x_0: np.ndarray,
                 omega: Optional[float] = None,
                 theta: np.ndarray = .1) -> None:
        self.dt = dt
        self.mu = mu
        self.sigma = sigma
        self.x = x_0
        self.theta = theta
        self.omega = omega
        self.j: Optional[np.ndarray] = None
        if self.omega is not None:
            assert self.x.shape[-1] == 2
            self.j = np.array([[0, 1],
                               [-1, 0]])

    def step(self) -> np.ndarray:
        """ One noise step. """
        delta_x = -(self.theta * (self.x - self.mu) * self.dt +
                    self.sigma * np.random.randn(*self.x.shape) * np.sqrt(self.dt))
        if self.omega is not None:
            delta_x += self.omega * self.dt * self.x @ self.j
        self.x += delta_x
        return self.x

    def reset(self, x_0: np.ndarray):
        """ Reset the process. """
        self.x = x_0

    def stationary(self, samples: int) -> np.ndarray:
        """ Samples from the stationary distribution. """
        omega_coef = 0 if self.omega is None else self.omega
        return (self.sigma / (2 * np.sqrt(self.theta ** 2 + omega_coef ** 2)) *
                np.random.randn(samples, self.x.shape[-1]))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.ion()
    T = 100000
    dt = .001
    mu = np.array([0., 0.])
    sigma = .3
    omega = .3
    x_0 = np.array([0., 0.])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    pline, = ax.plot([], [], alpha=.3)

    noise = OrnsteinUlhenbeck(dt, mu, sigma, x_0, omega, theta=1e-3)
    line: Tuple[List[float], List[float]] = ([], [])
    for t in range(T):
        line[0].append(noise.x[0])
        line[1].append(noise.x[1])
        noise.step()
        if t % int(1 / dt) == 0:
            pline.set_data(*line)
            plt.show()
            plt.pause(.01)
