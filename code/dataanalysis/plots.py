import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
from typing import List
from os.path import join


def plot_learning_curves(expdata, key_list: List[str], namefile: str,
                         mint: float = 0, maxt: float = 100, gtype: str = 'time_std'):
    MIN_DT = 5e-4
    MAX_DT = 5e-2
    plt.style.use('ggplot')
    algolabeldict = {
        'discrete_value': "qlearning",
        'discrete_advantage': "advup",
        'approximate_advantage': 'advup',
        'approximate_value': 'ddpg',
    }
    ref_dts = {
        "ant": .05,
        "bipedal_walker": .02,
        "cartpole": .02,
        "continuous_pendulum": .05,
        "half_cheetah": .05
    }
    dts = sorted(list(set([s.args['dt'] for s in expdata._settings])))
#    lss = [(0, ()), (0, (5, 1)), (0, (5, 5)), (0, (5, 10)), (0, (1, 5)), (0, (1, 10))]
#    dt_dict = {dt: ls for dt, ls in zip(dts, lss)}


    if len(dts) > 0:
        #        print(f'dts:{dts}')
        cnorm = matplotlib.colors.Normalize(vmin=np.log(MIN_DT) - 3, vmax=np.log(MAX_DT) + .1)
        cm1 = matplotlib.cm.ScalarMappable(norm=cnorm, cmap='Blues')
        cm2 = matplotlib.cm.ScalarMappable(norm=cnorm, cmap='Reds')

    nlines, ncol = len(key_list), 1
    fig, axes = plt.subplots(nlines, ncol, figsize=(5.*ncol, 4.*nlines))

    first = True
    for key, ax in zip(key_list, axes.flat if nlines > 1 else [axes]):
        # ax.set_title(key)

        for setting in sorted(expdata._settings, key=lambda s: s.args['dt']):
            args = setting.args
            dt = args['dt']
            nb_steps = args['nb_steps']
            nb_envs = args['nb_train_env']
            ref_dt = ref_dts[args['env_id']]
            tmint = mint * nb_steps * nb_envs / 3600
            tmaxt = maxt * nb_steps * nb_envs / 3600
            algo = args["algo"]
            label = f"{algolabeldict[algo]}; dt={dt:.0e}"
            # linestyle = dt_dict[dt]
            linestyle = '-'
            if 'value' in algo:
                # c = 'blue'
                c = cm1.to_rgba(np.log(dt))
                linewidth = 1.
                alpha = None
                if 'tau' in expdata.deltakeys:
                    tau = setting.args['tau']
                    label += f';tau{tau}'
            elif 'advantage' in algo:
                # c = 'red'
                c = cm2.to_rgba(np.log(dt))
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
                x = dt * nb_steps * nb_envs * x / 3600
                y = dt / ref_dt * ysmoothed
                sigma = dt * sigma / ref_dt

                if first:
                    ax.legend()
                ax.set_xlim(tmint, tmaxt)
                ax.set_xlabel("Physical time (hours)")
                ax.set_ylabel("Scaled return")
                ax.plot(x, y, label=label, c=c, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
                ax.fill_between(x, y-sigma, y+sigma, facecolor=c, alpha=0.2)
                # ax.plot(dt*xdata, dt*ydata, c=c, alpha=0.2, linestyle=linestyle, linewidth=linewidth)
            elif gtype == 'run_std':
                std_data = np.array([v for (i, v) in timeseq_stats['std'].items()])
                x = np.linspace(min(xdata), max(xdata), 400)
                y = interp1d(xdata, ydata, kind='cubic')(x)
                std = interp1d(xdata, std_data, kind='cubic')(x)
                x = dt * nb_steps * nb_envs * x / 3600
                y = y * dt / ref_dt
                std = std * dt / ref_dt
                ax.set_xlim(tmint, tmaxt)
                ax.set_xlabel("Physical time (hours)")
                ax.set_ylabel("Scaled return")
                ax.plot(x, y, label=label, c=c, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
                ax.fill_between(x, y - std, y + std, facecolor=c, alpha=0.2)
                # ax.fill_between(x, y - std / 2, y + std / 2, facecolor=c, alpha=0.2)
        first = False

    plt.tight_layout()
    # plt.savefig(namefile+'.eps', format="eps")
    plt.savefig(join('plots', namefile+'.png'), format='png', dpi=1000)
