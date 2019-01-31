import argparse
import os
import pickle
import re
from subprocess import call

import matplotlib.pyplot as plt


def plots(logs, namefile, key_regexp=".*(l|L)oss.*|Return"):
    fig, axes = plt.subplots(len(logs), figsize=(10., 15.))

    for logkey, ax in zip(logs, axes):
        if re.match(key_regexp, logkey) is None:
            continue
        ax.set_title(logkey)

        x = list(logs[logkey].keys())
        y = list(logs[logkey].values())

        ax.plot(x, y)
        ax.set_xlabel("timestamps")
        ax.set_xlim(0., max(x))
        l = len(x) # flake8: noqa
        ax.set_ylim(min(y[l//10:]), max(y[l//10:]))

    plt.tight_layout()
    print(f'Plot: {namefile}')
    plt.savefig(namefile, format="eps")


def summary(logs):
    print('Summary')
    metrics = [("Avg_adv_loss", min), ("Float_adv_loss", min), ("Return", max),
        ("loss/advantage", min), ("loss/policy", min), ]
    for met, f in metrics:
        if met not in logs:
            continue
        metval = logs[met]
        curr_ts, curr_val = max(metval.items(), key=lambda x: x[0])
        best_ts, best_val = f(metval.items(), key=lambda x: x[1])
        print(f"{met}\tBest value: {best_val:.2e} at ts {best_ts}\t Current value: {curr_val:.2e} at ts {curr_ts}\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitoring tool')
    subparsers = parser.add_subparsers(dest='command')

    parser.add_argument('--logdir', type=str,
                        help='Log directory')
    parser.add_argument('--sigma_eval', type=float, default=0.,
                        help='Sigma used for evaluation.')
    parser.add_argument('--plots', action='store_true', default=False,
                        help='plot metrics in logdir/plots.eps')
    parser.add_argument('--video', action='store_true',
                        help='outputs a video as a stack of numpy frames at '
                        'logdir/videos/demo_0.npz')
    parser.add_argument('--nolog', action='store_true',
                        help="don't compute log summaries")
    parser.add_argument('--xvfb', action='store_true', help='Use xvfb for rendering.')

    args = parser.parse_args()

    log_filename = os.path.join(args.logdir, 'logs.pkl')

    if args.video:
        if args.xvfb:
            cmd = ['xvfb-run', '-s', '"-screen 0 1400x900x24"']
        else:
            cmd = []
        cmd += ['python', 'eval.py', '--logdir', args.logdir, '--sigma_eval',
                str(args.sigma_eval), '--overwrite']
        print("Launching video rendering script: " + " ".join(cmd))
        call(' '.join(cmd), shell=True)

    if os.path.isfile(log_filename) and not args.nolog:
        with open(log_filename, 'rb') as f:
            logs = pickle.load(f)

        if args.plots:
            plots(logs, os.path.join(args.logdir, 'plots.eps'))
    else:
        raise ValueError(f'{log_filename} is not a file')

    summary(logs)