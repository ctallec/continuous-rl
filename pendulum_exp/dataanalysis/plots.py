import argparse
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import getpass


from dataloader import loader_leonard, ExperimentData

def plot_learning_curves(expdata, key_list, namefile):
    algolabeldict = {
        'discrete_value': "qlearning",
        'discrete_advantage': "advup",
        'approximate_value': "qlearning",
        'approximate_advantage': "advup"
    }

    mint, maxt = 0, 40

    nlines, ncol = len(key_list), 1
    fig, axes = plt.subplots(nlines, ncol, figsize=(5.*ncol, 4.*nlines))

    dtlist = [np.log(setting.args['dt']) for setting in expdata._setting_list]
    if len(dtlist) > 0:
        cnorm = matplotlib.colors.Normalize(vmin=min(dtlist), vmax=max(dtlist))
        cm = matplotlib.cm.ScalarMappable(norm=cnorm, cmap='plasma')

    first = True
    for key, ax in zip(key_list, axes.flat):
        ax.set_title(key)

        for setting in sorted(expdata._setting_list, key=lambda s: s.args['dt']):
            args = setting.args
            dt = args['dt']
            algo = args["algo"]
            label = f"{algolabeldict[algo]}; dt={dt:.1e}"
            if 'value' in algo:
                linestyle = '--'
                c = cm.to_rgba(np.log(dt))
                linewidth = 1.
                alpha = None
            elif 'advantage' in algo:
                linestyle = '-'
                c = cm.to_rgba(np.log(dt))
                linewidth = 1.
                alpha = None
            timeseq = setting.timeseq(key)
            if timeseq is None:
                continue

            xdata = np.array([i for (i, v) in timeseq.items()])
            ydata = np.array([v for (i, v) in timeseq.items()])

            x = np.arange(max(xdata)+1)
            yinterp = np.interp(x, xdata, ydata)
            ysmoothed = gaussian_filter1d(yinterp, sigma=0.3 / dt)

            sigma = np.sqrt(gaussian_filter1d((yinterp - ysmoothed) ** 2, sigma=0.3 / dt))
            x = dt * x
            y = dt * ysmoothed
            sigma = dt * sigma

            if first:
                ax.legend()
            ax.set_xlim(mint, maxt)
            ax.set_xlabel("time (second)")
            ax.plot(x, y, label=label, c=c, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
            ax.fill_between(x, y-sigma, y+sigma, facecolor=c, alpha=0.2)
            # ax.plot(dt*xdata, dt*ydata, c=c, alpha=0.2, linestyle=linestyle, linewidth=linewidth)
        first = False

    plt.tight_layout()
    plt.savefig(namefile+'.eps', format="eps")
    plt.savefig(namefile+'.png', format='png')


if __name__ == '__main__':
    usr = getpass.getuser()
    # start_date = datetime.strptime('2019_01_08_03_13_33', "%Y_%m_%d_%H_%M_%S")
    # stop_date = datetime.strptime('2019_01_08_03_13_35', "%Y_%m_%d_%H_%M_%S")
    start_date = 'last'
    stop_date = None
    if 'c' in usr:
        parser = argparse.ArgumentParser()
        parser.add_argument('--logdir', type=str, required=True)
        parser.add_argument('--exp_name', type=str, required=True)
        args = parser.parse_args()

        logdir = args.logdir
        exp_name = args.exp_name
    else:
        logdir = '/private/home/leonardb/workdir'
        exp_name = 'mujoco_continuous'
    runlist = loader_leonard(logdir, exp_name,
                             start_date=start_date, stop_date=stop_date)

    expdata = ExperimentData(runlist)

    plot_learning_curves(expdata, ['Return', 'Return'], 'out')
