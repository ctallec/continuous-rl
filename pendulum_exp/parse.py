"""Parsing facilities."""
import argparse
import argload

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='drau', choices=['drau', 'qlearn'])
    parser.add_argument('--dt', type=float, default=.05)
    parser.add_argument('--steps_btw_train', type=int, default=3)
    parser.add_argument('--steps_btw_catchup', type=int, default=10)
    parser.add_argument('--env_id', type=str, default='pendulum')
    parser.add_argument('--noise_type', type=str, default='parameter')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--nb_layers', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--nb_epochs', type=int, default=50000)
    parser.add_argument('--nb_steps', type=int, default=100)
    parser.add_argument('--sigma_eval', type=float, default=0)
    parser.add_argument('--sigma', type=float, default=1.5)
    parser.add_argument('--theta', type=float, default=7.5)
    parser.add_argument('--nb_train_env', type=int, default=32)
    parser.add_argument('--nb_eval_env', type=int, default=16)
    parser.add_argument('--memory_size', type=int, default=1000000)
    parser.add_argument('--learn_per_step', type=int, default=50)
    parser.add_argument('--cyclic_exploration', action='store_true')
    parser.add_argument('--normalize_state', action='store_true')
    parser.add_argument('--lr', type=float, default=.03)
    parser.add_argument('--policy_lr', type=float, default=None)
    parser.add_argument('--time_limit', type=float, default=None)
    parser.add_argument('--redirect_stdout', action='store_true')
    parser.add_argument('--nb_policy_samples', type=int, default=None)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'rmsprop'],
                        default='sgd')
    parser.add_argument('--noreload', action='store_true')
    parser = argload.ArgumentLoader(parser, to_reload=[
        'algo', 'dt', 'steps_btw_train', 'steps_btw_catchup', 'env_id', 'noise_type', 
        'batch_size', 'hidden_size', 'nb_layers', 'gamma', 'nb_epochs', 'nb_steps',
        'sigma_eval', 'sigma', 'theta', 'nb_train_env', 'nb_eval_env', 'memory_size',
        'learn_per_step', 'cyclic_expliration', 'normalize_state', 'lr', 'time_limit',
        'nb_policy_samples', 'policy_lr', 'alpha', 'beta', 'weight_decay', 'optimizer'
    ])
    return parser.parse_args()
