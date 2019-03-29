import argparse
from analog.load import ExperimentLog
from dataloader import loader
from plots import plot_learning_curves

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, required=True)
parser.add_argument('--exp_names', nargs='+', type=str, required=True)
parser.add_argument('--std_type', type=str, choices=['time', 'run'])
parser.add_argument('--min_t', type=float, default=0.)
parser.add_argument('--max_t', type=float, default=100.)
args = parser.parse_args()

start_date = 'last'
stop_date = None
expdata: ExperimentLog = ExperimentLog()
for exp_name in args.exp_names:
    expdata.extend(loader(args.logdir, exp_name, start_date=start_date, stop_date=stop_date))

# def filter(args) -> bool:
#     return ('noscale' in args and args['noscale'] and 'value' in args['algo']) \
#         or 'advantage' in args['algo']


# expdata = filter(expdata, filter)

plot_learning_curves(expdata, ['Return'], args.exp_names[0], mint=args.min_t, maxt=args.max_t, gtype=args.std_type + "_std")
expdata.repr_rawlogs("Return", 5)
