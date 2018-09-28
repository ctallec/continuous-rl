""" Generates trajectories. """
from collections import namedtuple
from noise import OrnsteinUlhenbeck
import numpy as np

OrnsteinUlhenbeckParameters = namedtuple(
    'OrnsteinUlhenbeckParameters',
    ('dt', 'sigma', 'theta'))

def generate_trajectories(
        batch_size: int,
        T: int,
        w: float,
        ornstein_ulhenbeck_params: OrnsteinUlhenbeckParameters) -> np.ndarray:
    """ Generate trajectories. """
    # create corresponding ornstein_ulhenbeck processes
    noise = OrnsteinUlhenbeck(
        ornstein_ulhenbeck_params.dt,
        np.zeros((1, 1)),
        ornstein_ulhenbeck_params.sigma,
        np.zeros((batch_size, 2)),
        w, ornstein_ulhenbeck_params.theta)

    noise.reset(noise.stationary(batch_size))

    noises = np.zeros((batch_size, T, 2))
    for t in range(T):
        noises[:, t] = noise.x
        noise.step()
    return noises, noise

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = generate_trajectories(
        1, 100000, .3,
        OrnsteinUlhenbeckParameters(.01, .1, .01))

    plt.plot(data[0, :, 0], data[0, :, 1])
    plt.show()
