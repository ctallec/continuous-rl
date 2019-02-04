"""Parsing facilities."""
import argparse
import argload

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_gap', type=float, default=.1,
                        help='evaluation is performed every .1/dt epochs.')
    parser.add_argument('--algo', type=str, default='approximate_advantage',
                        help='algorithm used.')
    parser.add_argument('--dt', type=float, default=.02,
                        help='temporal discretization.')
    parser.add_argument('--steps_btw_train', type=int, default=10,
                        help='number of environment steps between two training periods.')
    parser.add_argument('--env_id', type=str, default='pendulum',
                        help='environment.')
    parser.add_argument('--noise_type', type=str, default='action', choices=['action', 'parameter'],
                        help='noise type used (parameter is ill-behaved)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='number of hidden units per layer.')
    parser.add_argument('--nb_layers', type=int, default=1,
                        help='number of layers (careful, the "true number of layers" is this number + 1).')
    parser.add_argument('--gamma', type=float, default=.8,
                        help='discount factor.')
    parser.add_argument('--nb_true_epochs', type=float, default=50,
                        help='number of true epochs (epochs / dt) to train on.')
    parser.add_argument('--nb_steps', type=int, default=100,
                        help='number of environment steps in an epoch')
    parser.add_argument('--sigma', type=float, default=1.5,
                        help='OU noise parameter.')
    parser.add_argument('--theta', type=float, default=7.5,
                        help='OU stiffness parameter.')
    parser.add_argument('--c_entropy', type=float, default=1e-4,
                        help='entropy regularization')
    parser.add_argument('--nb_train_env', type=int, default=32,
                        help='number of parallel environments during training.')
    parser.add_argument('--nb_eval_env', type=int, default=16,
                        help='number of parallel environments used to evaluate.')
    parser.add_argument('--memory_size', type=int, default=1000000,
                        help='size of the memory buffer.')
    parser.add_argument('--learn_per_step', type=int, default=50,
                        help='number of gradient step in one learning step')
    parser.add_argument('--normalize_state', action='store_true',
                        help='is state normalization used.')
    parser.add_argument('--lr', type=float, default=.03,
                        help='critic learning rate.')
    parser.add_argument('--policy_lr', type=float, default=None,
                        help='policy learning rate (for approximate policies).')
    parser.add_argument('--time_limit', type=float, default=None,
                        help='specify environment time limite (physical time).')
    parser.add_argument('--redirect_stdout', action='store_true',
                        help='should we redirect stdout to a log file?')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='actor weight decay.')
    parser.add_argument('--alpha', type=float, default=None,
                        help='prioritized replay buffer alpha (untested).')
    parser.add_argument('--beta', type=float, default=None,
                        help='prioritized replay buffer beta (untested).')
    parser.add_argument('--tau', type=float, default=.99,
                        help='target network update rate (works for all '
                        'algo, do not expect it to work with dau).')
    parser.add_argument('--noscale', action='store_true',
                        help='use unscaled ddpg when set')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'rmsprop', 'adam'],
                        default='sgd')
    parser.add_argument('--noreload', action='store_true',
                        help='do not reload previously saved model when set.')
    parser = argload.ArgumentLoader(parser, to_reload=[
        'algo', 'dt', 'steps_btw_train', 'env_id', 'noise_type',
        'batch_size', 'hidden_size', 'nb_layers', 'gamma', 'nb_true_epochs', 'nb_steps',
        'sigma', 'theta', 'c_entropy', 'nb_train_env', 'nb_eval_env', 'memory_size',
        'learn_per_step', 'normalize_state', 'lr', 'time_limit',
        'policy_lr', 'alpha', 'beta', 'weight_decay', 'optimizer',
        'tau', 'eval_gap', 'noscale'
    ])
    args = parser.parse_args()

    # args translation
    if args.algo == 'ddpg':
        args.algo = 'approximate_value'
    elif args.algo == 'dqn':
        args.algo = 'discrete_value'
    elif args.algo == 'ddau':
        args.algo = 'discrete_advantage'
    elif args.algo == 'cdau':
        args.algo = 'approximate_advantage'

    return args
