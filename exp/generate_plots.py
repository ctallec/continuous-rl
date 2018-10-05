""" Generate various plots. """
import numpy as np
import matplotlib.pyplot as plt

def generate_images(nb_figure: int): # pylint: disable=too-many-locals
    """ Generate images. """
    fontsize = 10
    fig = plt.figure(nb_figure)

    lindex = (0, 3000, 9000, 15000, 27000)
    span = 1 / len(lindex)

    x = span / 2
    for idx in lindex:
        plt.text(x, 0, idx, fontsize=fontsize, horizontalalignment='center')
        x += span
    plt.arrow(0, 0, 1, 0)

    nb_indices = len(lindex)
    dts = [1e-1, 1e-2, 3e-3]
    nb_dts = len(dts)

    yspan = 1 / nb_dts
    y = yspan / 2
    for yidx in dts:
        plt.text(-.05, y, yidx, fontsize=fontsize, verticalalignment='center', horizontalalignment='center')
        y += yspan

    for i, dt in enumerate(dts):
        suffix = f"{dt:1.0E}".replace('E', 'e').replace('0', '')
        for k, t in enumerate(lindex):
            fig.add_subplot(3, nb_indices, 1 + nb_indices * i + k)
            sub_fig = plt.imshow(np.load(f'logs/dt_{suffix}/V_{t}.npy'))
            sub_fig.set_cmap('plasma')
            sub_fig.axes.get_xaxis().set_visible(False)
            sub_fig.axes.get_yaxis().set_visible(False)
    fig.axes[0].axis('off')
    fig.tight_layout(w_pad=0.3, h_pad=.3)

def generate_curve(nb_figure: int):
    """ Generate curves. """
    fig = plt.figure(nb_figure)
    colors = ['#fbb4ae', '#b3cde3', '#ccebc5']
    for dt, c in zip([1e-1, 1e-2, 3e-3], colors):
        suffix = f"{dt:1.0E}".replace('E', 'e').replace('0', '')
        x = np.loadtxt(f'logs/dt_{suffix}/eval.log')
        plt.plot(x[:, 0], x[:, 2] * dt, label=f'dt {suffix}')
        plt.legend()

if __name__ == '__main__':
    generate_images(1)
    plt.savefig('../imgs/V_functions.pdf')
    generate_curve(2)
    plt.savefig('../imgs/learning_curve.pdf')
    input()
