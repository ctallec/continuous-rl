import argparse
import os
import pickle

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Monitoring tool')
subparsers = parser.add_subparsers(dest='command')

parser.add_argument('--logdir', type=str, 
                    help='Log directory')

parser.add_argument('--plots', action='store_true', default=False,
    help='plot metrics in logdir/plots.eps')


def plots(logs, namefile):
    fig, axes = plt.subplots(len(logs), figsize=(10.,15.))

    for logkey, ax in zip(logs, axes):
        ax.set_title(logkey)

        x = list(logs[logkey].keys())
        y = list(logs[logkey].values())

        ax.plot(x, y)
        ax.set_xlabel("timestamps")
        ax.set_xlim(0., max(x))
        l = len(x)
        ax.set_ylim(min(y[l//10:]), max(y[l//10:]))

    plt.tight_layout()
    print(f'Plot: {namefile}')
    plt.savefig(namefile, format="eps")


def summary(logs):
    print('Summary')
    metrics = [("Avg_adv_loss", min), ("Return", max)]
    for met, f in metrics:
        metval = logs[met]
        curr_ts, curr_val = max(metval.items(), key=lambda x: x[0])
        best_ts, best_val = f(metval.items(), key=lambda x: x[1])
        print(f"{met}\tBest value: {best_val:.2e} at ts {best_ts}\t Current value: {curr_val:.2e} at ts {curr_ts}\t")


if __name__ == '__main__':
    args = parser.parse_args()

    log_filename = os.path.join(args.logdir, 'logs.pkl')
    assert os.path.isfile(log_filename)
    with open(log_filename, 'rb') as f:
        logs = pickle.load(f)

    if args.plots:
        plots(logs, os.path.join(args.logdir, 'plots.eps'))

    summary(logs)