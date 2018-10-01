""" Generate various plots. """
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for dt in [1e-1, 3e-2, 1e-2, 3e-3]:
        suffix = f"{dt:1.0E}".replace('E', 'e').replace('0', '')
        x = np.loadtxt(f'logs/dt_{suffix}/eval.log')
        plt.plot(x[:, 0], x[:, 2] * dt, label=f'dt {suffix}')
    plt.show()
    input()
