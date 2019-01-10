import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
from scipy.ndimage.filters import gaussian_filter1d


from dataloader import loader_leonard, ExperimentData
from datetime import datetime

algolabeldict = {
    'discrete_value':"qlearning",
    'discrete_advantage':"advup",
    'approximate_advantage':'advup',
    'approximate_value':'ddpg',
}

def plot_learning_curves(expdata, key_list, namefile):
    mint, maxt = 0, 100

    nlines, ncol = len(key_list), 1
    fig, axes = plt.subplots(nlines, ncol, figsize=(5.*ncol, 4.*nlines))

    dtlist = [np.log(setting.args['dt']) for setting in expdata._setting_list]
    if len(dtlist) > 0:
        cnorm = matplotlib.colors.Normalize(vmin=min(dtlist), vmax=max(dtlist))
        cm = matplotlib.cm.ScalarMappable(norm=cnorm, cmap='plasma')

    first = True
    for key, ax in zip(key_list, axes.flat):
        ax.set_title(key)

        for setting in sorted(expdata._setting_list, key=lambda s:s.args['dt']):
            args = setting.args
            dt = args['dt']
            algo = args["algo"]
            label = f"{algolabeldict[algo]}; dt={dt:.0e}"
            if algo == 'discrete_value' or algo == 'approximate_value':
                linestyle = '--'
                c=cm.to_rgba(np.log(dt))
                linewidth=1.
                alpha=None
                if 'tau' in expdata.deltakeys:
                    tau = setting.args['tau']
                    label += f';tau{tau}'
                    if tau != 0:
                        linestyle = ':'
            elif algo == 'discrete_advantage' or algo == 'approximate_advantage':
                linestyle = '-'
                c=cm.to_rgba(np.log(dt))
                linewidth=1.
                alpha = None
            else:
                raise ValueError
            timeseq = setting.timeseq(key)
            if timeseq is None:
                continue

            xdata = np.array([i for (i,v) in timeseq.items()])
            ydata = np.array([v for (i,v) in timeseq.items()])

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
            ax.set_xlabel("time (second)")
            ax.plot(x, y, label=label, c=c, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
            ax.fill_between(x, y-sigma, y+sigma, facecolor=c, alpha=0.2)
            # ax.plot(dt*xdata, dt*ydata, c=c, alpha=0.2, linestyle=linestyle, linewidth=linewidth)
        first = False

    plt.tight_layout()
    # plt.savefig(namefile+'.eps', format="eps")
    plt.savefig(namefile+'.png', format='png', dpi=1000)

start_date = datetime.strptime('2019_01_09_05_44_37', "%Y_%m_%d_%H_%M_%S")
# stop_date = datetime.strptime('2019_01_08_03_13_35', "%Y_%m_%d_%H_%M_%S")
# start_date = 'last'
stop_date = None
runlist = loader_leonard('/private/home/leonardb/workdir', 'mujoco_continuous', 
    start_date=start_date, stop_date=stop_date)

expdata = ExperimentData(runlist)

print(expdata.deltakeys)

expdata_ant = expdata.filter_settings(lambda s: s['env_id'] == 'ant')
plot_learning_curves(expdata_ant, ['Return', 'Return'], 'ant')

expdata_cartpole = expdata.filter_settings(lambda s: s['env_id'] == 'cartpole')
plot_learning_curves(expdata_cartpole, ['Return', 'Return'], 'cartpole')


# expdata_tau = expdata.filter_settings(lambda s: s['algo'] == 'discrete_value')
# expdata_tau.repr_rawlogs("Return", 5)
























