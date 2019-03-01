from datetime import datetime

from analog.load import ExperimentLog, filter
from dataloader import loader
from plots import plot_learning_curves

# 2019_01_14_10_22_54//
start_date = datetime.strptime('2019_01_14_10_22_54', "%Y_%m_%d_%H_%M_%S")
# stop_date = datetime.strptime('2019_01_21_00_39_51', "%Y_%m_%d_%H_%M_%S")

# start_date = 'last'
stop_date = None
runlist = loader('/private/home/leonardb/workdir', 'mujoco_continuous',
                 start_date=start_date, stop_date=stop_date)

expdata = ExperimentLog(runlist)


def predicat_noscale(s):
    return ('noscale' in s and s['noscale'])


scaled = True

expdata = filter(expdata, lambda s: (predicat_noscale(s) != scaled) or 'advantage' in s['algo'])

if scaled:
    suffix = '_scaled'
else:
    suffix = '_unscaled'


expdata_ant = filter(expdata, lambda s: s['env_id'] == 'ant')
plot_learning_curves(expdata_ant, ['Return'], 'ant'+suffix, gtype='run_std', mint=0, maxt=20, maxrun=5)


expdata_cheetah = filter(expdata, lambda s: s['env_id'] == 'half_cheetah')
expdata_cheetah = filter(expdata_cheetah, lambda s: s['nb_true_epochs'] == 50)
plot_learning_curves(expdata_cheetah, ['Return'], 'cheetah'+suffix, gtype='run_std', mint=0, maxt=20)

expdata_bipedal = filter(expdata, lambda s: s['env_id'] == 'bipedal_walker')
plot_learning_curves(expdata_bipedal, ['Return'], 'bipedal_walker'+suffix, gtype='run_std', mint=0, maxt=2.5)

# expdata_cartpole = expdata.filter(lambda s: s['env_id'] == 'cartpole')
# plot_learning_curves(expdata_cartpole, ['Return', 'Return'], 'cartpole', gtype='run_std', mint=0, maxt=20)


# expdata_tau = expdata.filter(lambda s: s['algo'] == 'discrete_value')
# expdata_tau.repr_rawlogs("Return", 5)
