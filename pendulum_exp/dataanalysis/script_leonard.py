from datetime import datetime



from dataloader import loader_leonard, ExperimentData

from plots import plot_learning_curves

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

expdata_cheetah = expdata.filter_settings(lambda s: s['env_id'] == 'half_cheetah')
plot_learning_curves(expdata_cheetah, ['Return', 'Return'], 'cheetah')


expdata_tau = expdata.filter_settings(lambda s: s['algo'] == 'discrete_value')
expdata_tau.repr_rawlogs("Return", 5)
