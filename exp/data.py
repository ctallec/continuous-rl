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
    # randomly samples elements on the unit circle
    theta = np.random.uniform(-np.pi, np.pi, (batch_size,))
    thetas = np.ones((batch_size, T)) * w * ornstein_ulhenbeck_params.dt
    thetas[:, 0] = theta
    thetas = np.cumsum(thetas, axis=1)
    xs = np.stack([np.cos(thetas), np.sin(thetas)], axis=2)
    sigma = np.array(ornstein_ulhenbeck_params.sigma).reshape(1, 1)

    # create corresponding ornstein_ulhenbeck processes
    noise = OrnsteinUlhenbeck(
        ornstein_ulhenbeck_params.dt,
        np.zeros((1, 1)),
        sigma,
        np.zeros((batch_size, 2)),
        ornstein_ulhenbeck_params.theta)

    noises = np.zeros((batch_size, T, 2))
    for t in range(T):
        noises[:, t] = noise.x
        noise.step()
    return xs + noises

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = generate_trajectories(
        1, 100000, .2,
        OrnsteinUlhenbeckParameters(.01, .1, .1))

    plt.plot(data[0, :, 0], data[0, :, 1])
    plt.show()
