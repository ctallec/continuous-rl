""" Obtain optimal v function using FEM. """
from skfem import bilinear_form, linear_form
from skfem import InteriorBasis
from skfem import ElementTriP1
from skfem import asm, condense, solve
import numpy as np
from fem_utils import cylindrical_mesh

def angle_normalize(x: np.ndarray) -> np.ndarray:
    """ Normalize angle. """
    return ((x+np.pi) % (2*np.pi)) - np.pi

def reward(s: np.ndarray) -> np.ndarray:
    """ Returns reward. """
    return - angle_normalize(s[0]) ** 2 - .1 * s[1] ** 2


class PendulumUtilities:
    """ Pendulum utilities. """
    def __init__(self, g: float, l: float, m: float, dt: float) -> None:
        self.g = g
        self.l = l
        self.m = m
        self.dt = dt

    def state_transition(self, s: np.ndarray,
                         u: np.ndarray) -> np.ndarray:
        """ Pendulum state transition. """
        g = self.g
        l = self.l
        m = self.m
        dth = s[1]
        ddth = (- 3 * g / (2 * l) * np.sin(s[0] + np.pi) +
                3 / (m * l ** 2) * u)
        return np.stack(
            [dth, ddth], axis=0)

    def max_transition(self, s: np.ndarray, du: np.ndarray):
        """ take max_u(r(s, u) + V(s')). """
        return np.maximum(*[(du * self.state_transition(s, k)).sum(axis=0) for k in (-1, 1)])


def solve_exact(pendulum_utility: PendulumUtilities):
    """ Solves Optimal Bellman equation using FEM. """

    th_scale = 2 * np.pi
    dth_scale = 1
    m = cylindrical_mesh(
        (th_scale, dth_scale),
        (-th_scale / 2, -dth_scale / 2),
        6)

    gamma = .9
    sigma = .1

    e = ElementTriP1()
    basis = InteriorBasis(m, e)

    @bilinear_form
    def optimal_bellman(u, du, v, dv, w): # pylint: disable=unused-argument
        return pendulum_utility.max_transition(w[0], du) * v - sigma ** 2 / 2 * np.sum(du * dv, axis=0)

    @linear_form
    def reward_form(v, dv, w): # pylint: disable=unused-argument
        return -v * reward(w[0])

    A = asm(optimal_bellman, basis)
    b = asm(reward_form, basis)

    I = m.interior_nodes()
    x = 0 * b
    x[I] = solve(*condense(A, b, I=I))
#    f = m.interpolator(x)

    return m, x

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    pendulum_utils = PendulumUtilities(g=10., l=1., m=1., dt=.05)
    m, x = solve_exact(pendulum_utils)
    m.plot3(x)
    input()
    plt.savefig('logs/misc.pdf')
