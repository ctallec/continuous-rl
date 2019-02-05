from typing import Tuple
from functools import partial
from policies.offline_policy import OfflinePolicy
from envs.env import Env
from actors import DiscreteActor, ApproximateActor
from critics import AdvantageCritic, ValueCritic
from envs.utils import make_env
from envs.vecenv import VEnv
from noises.setupnoise import setup_noise
from policies.policy import Policy
from policies.a2c import A2CPolicy
from actors.a2cactor import A2CActor
from critics.a2ccritic import A2CCritic


def configure(args) -> Tuple[Policy, Env, Env]:
    env_fn = partial(make_env, env_id=args.env_id,
                     dt=args.dt, time_limit=args.time_limit) 

    env: Env = VEnv([env_fn() for _ in range(args.nb_train_env)])
    eval_env: Env = VEnv([env_fn() for _ in range(args.nb_eval_env)])

    if args.algo in ["approximate_value", "approximate_advantage", "discrete_value, discrete_adavantage"]:
        noise = setup_noise(
            noise_type=args.noise_type, sigma=args.sigma,
            theta=args.theta, dt=args.dt, sigma_decay=lambda _: 1.,
            action_space=eval_env.action_space, noscale=args.noscale)

        actor_type, critic_type = args.algo.split('_')

        kwargs = dict(
            dt=args.dt, gamma=args.gamma, lr=args.lr, optimizer=args.optimizer,
            action_space=eval_env.action_space, observation_space=eval_env.observation_space,
            nb_layers=args.nb_layers, hidden_size=args.hidden_size,
            normalize=args.normalize_state, weight_decay=args.weight_decay, noise=noise,
            tau=args.tau, noscale=args.noscale
        )

        critic_cls = {
            "advantage": AdvantageCritic,
            "value": ValueCritic,
        }[critic_type]

        critic = critic_cls.configure(**kwargs)
        kwargs["critic_function"] = critic.critic_function()
        kwargs["target_critic_function"] = critic.critic_function(target=True)

        actor_cls = {
            "approximate": ApproximateActor,
            "discrete": DiscreteActor}[actor_type]

        actor = actor_cls.configure(**kwargs)

        policy = OfflinePolicy(
            steps_btw_train=args.steps_btw_train, learn_per_step=args.learn_per_step,
            memory_size=args.memory_size,
            batch_size=args.batch_size, alpha=args.alpha, beta=args.beta,
            actor=actor, critic=critic)
    elif args.algo == "a2c":

        actor = A2CActor.configure(
            action_space=eval_env.action_space, observation_space=eval_env.observation_space,
            nb_layers=args.nb_layers, hidden_size=args.hidden_size,
            lr=args.policy_lr, tau=args.tau, optimizer=args.optimizer, dt=args.dt, c_entropy=args.c_entropy,
            weight_decay=args.weight_decay
            )
        critic = A2CCritic.configure(
            dt=args.dt, gamma=args.gamma, lr=args.lr, optimizer=args.optimizer,
            observation_space=eval_env.observation_space,
            nb_layers=args.nb_layers, hidden_size=args.hidden_size,
            tau=args.tau, noscale=args.noscale) 

        policy = A2CPolicy(args.memory_size, args.batch_size, args.n_step,
            args.steps_btw_train, args.learn_per_step, args.nb_train_env,
            actor, critic)
        
    else:
        raise ValueError(f"Unknown algorithm {args.algo}")



    return policy, env, eval_env
