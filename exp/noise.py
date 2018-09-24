""" Define noise model. """
from typing import List, Tuple
import numpy as np

class OrnsteinUlhenbeck(object): # pylint: disable=too-few-public-methods
    """ Ornstein Ulhenbeck process. """
    def __init__(self, # pylint: disable=too-many-arguments
                 dt: float,
                 mu: np.ndarray,
                 sigma: np.ndarray,
                 x_0: np.ndarray,
                 theta: np.ndarray=.1) -> None:
        self.dt = dt
        self.mu = mu
        self.sigma = sigma
        self.x = x_0
        self.theta = theta

    def step(self) -> np.ndarray:
        """ One noise step. """
        delta_x = -(self.theta * (self.x - self.mu) * self.dt +
                    self.sigma * np.random.randn(*self.x.shape) * np.sqrt(self.dt))
        self.x += delta_x
        return self.x

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.ion()
    T = 100000
    dt = .001
    mu = np.array([1., 1.])
    sigma = .1
    x_0 = np.array([0., 0.])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-1, 3])
    ax.set_ylim([-1, 3])
    pline, = ax.plot([], [], alpha=.3)

    noise = OrnsteinUlhenbeck(dt, mu, sigma, x_0)
    line: Tuple[List[float], List[float]] = ([], [])
    for t in range(T):
        line[0].append(noise.x[0])
        line[1].append(noise.x[1])
        noise.step()
        if t % int(1 / dt) == 0:
            pline.set_data(*line)
            plt.show()
            plt.pause(.01)
