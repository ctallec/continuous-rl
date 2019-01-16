import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
from typing import List


def plot_learning_curves(expdata, key_list: List[str], namefile: str,
                         mint: float = 0, maxt: float = 100, gtype: str = 'time_std'):
    algolabeldict = {
        'discrete_value': "qlearning",
        'discrete_advantage': "advup",
        'approximate_advantage': 'advup',
        'approximate_value': 'ddpg',
    }
    dts = sorted(list(set([s.args['dt'] for s in expdata._settings])))
    lss = [(0, ()), (0, (5, 1)), (0, (5, 5)), (0, (5, 10)), (0, (1, 5)), (0, (1, 10))]
    dt_dict = {dt: ls for dt, ls in zip(dts, lss)}

    nlines, ncol = len(key_list), 1
    fig, axes = plt.subplots(nlines, ncol, figsize=(5.*ncol, 4.*nlines))

    first = True
    for key, ax in zip(key_list, axes.flat):
        ax.set_title(key)

        for setting in sorted(expdata._settings, key=lambda s: s.args['dt']):
            args = setting.args
            dt = args['dt']
            algo = args["algo"]
            label = f"{algolabeldict[algo]}; dt={dt:.0e}"
            linestyle = dt_dict[dt]
            if 'value' in algo:
                c = 'blue'
                linewidth = 1.
                alpha = None
                if 'tau' in expdata.deltakeys:
                    tau = setting.args['tau']
                    label += f';tau{tau}'
            elif 'advantage' in algo:
                c = 'red'
                linewidth = 1.
                alpha = None
            else:
                raise ValueError

            timeseq_stats = setting.timeseq(key)
            if timeseq_stats is None:
                continue
            timeseq = timeseq_stats['mean']

            xdata = np.array([i for (i, v) in timeseq.items()])
            ydata = np.array([v for (i, v) in timeseq.items()])

            if gtype == 'time_std':
                kernelsize = max(maxt/100, 0.3) / dt
                x = np.arange(max(xdata)+1)
                yinterp = np.interp(x, xdata, ydata)
                ysmoothed = gaussian_filter1d(yinterp, sigma=kernelsize)

                sigma = np.sqrt(gaussian_filter1d((yinterp - ysmoothed) ** 2, sigma=kernelsize))
                x = dt * x
                y = dt * ysmoothed
                sigma = dt * sigma

                if first:
                    ax.legend()
                ax.set_xlim(mint, maxt)
                ax.set_xlabel("Physical time")
                ax.set_ylabel("Scaled return")
                ax.plot(x, y, label=label, c=c, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
                ax.fill_between(x, y-sigma, y+sigma, facecolor=c, alpha=0.2)
                # ax.plot(dt*xdata, dt*ydata, c=c, alpha=0.2, linestyle=linestyle, linewidth=linewidth)
            elif gtype == 'run_std':
                std_data = np.array([v for (i, v) in timeseq_stats['std'].items()])
                x = np.linspace(min(xdata), max(xdata), 400)
                y = interp1d(xdata, ydata, kind='cubic')(x)
                std = interp1d(xdata, std_data, kind='cubic')(x)
                x = x * dt
                y = y * dt
                std = std * dt
                ax.set_xlim(mint, maxt)
                ax.set_xlabel("Physical time")
                ax.set_ylabel("Scaled return")
                ax.plot(x, y, label=label, c=c, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
                ax.fill_between(x, y - std / 2, y + std / 2, facecolor=c, alpha=0.2)
        first = False

    plt.tight_layout()
    # plt.savefig(namefile+'.eps', format="eps")
    plt.savefig(namefile+'.png', format='png', dpi=1000)
