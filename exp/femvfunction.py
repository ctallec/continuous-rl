# pylint: disable=too-many-locals, too-many-arguments
""" Attempt at FEMing bellman. """
from typing import Tuple
import numpy as np
from skfem import MeshTri, InteriorBasis
from skfem import ElementTriP1
from skfem import bilinear_form, linear_form
from skfem import asm, solve, condense

def solve_exact(sigma: float, theta: float, omega: float,
                beta: float, gamma: float, thresh: float,
                state_lim: Tuple[float, float]):
    """ Solve Bellman equation directly using FEM. """
    m = MeshTri()
    scale = state_lim[1] - state_lim[0]
    m.scale(scale)
    m.translate([state_lim[0], state_lim[0]])
    m.refine(6)

    e = ElementTriP1()
    basis = InteriorBasis(m, e)
    mu = 0

    J = np.array(
        [[0, -1],
         [1, 0]])

    transition_matrix = omega * J - theta * np.eye(2)

    @bilinear_form
    def bellman_form(u, du, v, dv, w):
        """ Laplace operator. """
        laplace = -sigma ** 2 / 2 * sum(du * dv)
        shape = w.x.shape
        state_trans = (transition_matrix @ (w.x.reshape(2, -1) - mu)).reshape(shape)
        next_state_term = sum(du * state_trans) * v
        state_term = (1 - gamma) * u * v
        return laplace + next_state_term - state_term


    def reward(x):
        """ Reward function. """
        threshold = (np.sum(x ** 2, axis=0) < thresh ** 2).astype(np.float32)
        return (1 / (1 + np.exp(- beta * x[0])) - .5)* threshold

    @linear_form
    def reward_form(v, dv, w): # pylint: disable=unused-argument
        return -v * reward(w.x)

    A = asm(bellman_form, basis)
    b = asm(reward_form, basis)

    I = m.interior_nodes()
    x = 0*b
    x[I] = solve(*condense(A, b, I=I))
    f = m.interpolator(x)

    return m, x, f

if __name__ == '__main__':
    sigma = .3
    theta = .01
    mu = 0
    omega = .1
    beta = 30
    gamma = .9
    reward_state_threshold = 2
    m, x, f = solve_exact(sigma, theta, omega, beta, gamma,
                          reward_state_threshold, state_lim=(-5, 5))
    m.plot(x)
    print(f([1, 2, 3], [1, 2, 3]))
    input()
