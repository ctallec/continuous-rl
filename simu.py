""" Simple simulations to validate theory """
import numpy as np
import matplotlib.pyplot as plt

# settable
deltat = .1
gamma = .9
alpha = .5
reward_function = 'dirac'
policy_type = 'white'

# derived
true_gamma = (1 - (1 - gamma) * deltat)
true_alpha = alpha * deltat

def associate_policy(policy_type):
    """ Returns a policy given a policy type """
    action = None # pylint: disable=unused-variable

    if policy_type == 'white':
        def _policy(state): # pylint: disable=unused-argument
            return np.random.randint(0, 2) * 2 - 1

    return _policy

policy = associate_policy(policy_type)

# env specific
ub = 1
lb = -1
state_space = np.arange(lb, ub + deltat, deltat)
V = np.zeros_like(state_space)

reward_table = [{
    'quadratic': lambda x: - x ** 2,
    'dirac': lambda x: 1 if np.abs(x) < deltat / 2 else 0
}[reward_function](x) for x in state_space]


def index_from_state(state):
    """ return index of state from state """
    return int(state / deltat + 1 / deltat)

def transition(state, action):
    """ transitions from state action """
    next_state = state + action * deltat
    if next_state > ub or next_state < lb:
        next_state = state - action * deltat
    return next_state, reward_table[index_from_state(next_state)]

state = 0
step = 0
traj = []

plt.ion()
while True:
    action = policy(state)
    next_state, reward = transition(state, action)
    traj += [next_state]
    index = index_from_state(state)
    next_index = index_from_state(next_state)
    V[index] = (1 - true_alpha) * V[index] + \
        true_alpha * (reward + true_gamma * V[next_index])
    state = next_state
    if step % 100 == 0:
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(traj)
        plt.subplot(2, 1, 2)
        plt.plot(V)
        plt.show()
        plt.pause(.1)
    step += 1

plt.plot(state_space, reward)
plt.show()
