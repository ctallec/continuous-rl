import argparse
from dataloader import loader_leonard, ExperimentData
from plots import plot_learning_curves

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, required=True)
parser.add_argument('--exp_name', type=str, required=True)
args = parser.parse_args()

start_date = 'last'
stop_date = None
runlist = loader_leonard(args.logdir, args.exp_name, start_date=start_date, stop_date=stop_date)

expdata = ExperimentData(runlist)

plot_learning_curves(expdata, ['Return', 'Return'], 'pendulum')
