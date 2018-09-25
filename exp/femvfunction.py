""" Attempt at FEMing bellman. """
import numpy as np
from skfem import MeshTri, InteriorBasis
from skfem import ElementTriP1
from skfem import bilinear_form, linear_form
from skfem import asm, solve, condense

m = MeshTri()
scale = 10
m.scale(scale)
m.translate([-scale / 2, -scale / 2])
m.refine(6)

e = ElementTriP1()
basis = InteriorBasis(m, e)

sigma = .3
theta = .01
mu = 0
omega = .1
beta = 30
gamma = .01
reward_state_threshold = 2
J = np.array(
    [[0, 1],
     [-1, 0]])

@bilinear_form
def bellman_form(u, du, v, dv, w):
    """ Laplace operator. """
    laplace = sigma ** 2 / 2 * sum(du * dv)
    shape = w.x.shape
    state_trans = theta * (w.x - mu) + omega * (J @ w.x.reshape(2, -1)).reshape(*shape)
    next_state_term = - sum(du * state_trans) * v
    state_term = - (1 - gamma) * u * v
    return -laplace + next_state_term + state_term


def reward(x):
    return 1 / (1 + np.exp(- beta * x[0])) * (np.sum(x ** 2, axis=0) < reward_state_threshold ** 2).astype(np.float32)

@linear_form
def reward_form(v, dv, w):
    return -v * reward(w.x)

A = asm(bellman_form, basis)
b = asm(reward_form, basis)

I = m.interior_nodes()
x = 0*b
x[I] = solve(*condense(A, b, I=I))

if __name__ == "__main__":
    m.plot(x)
    f = m.interpolator
    m.show()
    input()
