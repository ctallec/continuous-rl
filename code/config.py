"""Configuration facilities."""
from typing import Tuple
from functools import partial
from envs.env import Env
from actors import DiscreteActor, ApproximateActor
from critics import AdvantageCritic, ValueCritic
from envs.utils import make_env
from envs.vecenv import VEnv
from noises.setup import setup_noise
from agents.agent import Agent
from agents.off_policy.offline_agent import OfflineAgent
from agents.on_policy.a2c import A2CAgent
from agents.on_policy.ppo import PPOAgent
from actors.on_policy.a2c import A2CActor
from actors.on_policy.ppo import PPOActor
from critics.on_policy.ppo import PPOCritic
from critics.on_policy.a2c import A2CCritic

def configure(args) -> Tuple[Agent, Env, Env]:
    """
    Takes argparse args and generates the corresponding
    agent, environment and evaluation environment.
    """
    env_fn = partial(make_env, env_id=args.env_id,
                     dt=args.dt, time_limit=args.time_limit)

    env: Env = VEnv([env_fn() for _ in range(args.nb_train_env)])
    eval_env: Env = VEnv([env_fn() for _ in range(args.nb_eval_env)])

    if args.algo in ["approximate_value", "approximate_advantage",
                     "discrete_value", "discrete_advantage"]:
        noise = setup_noise(
            noise_type=args.noise_type, sigma=args.sigma, epsilon=args.epsilon,
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

        critic = critic_cls.configure(**kwargs) # type: ignore
        kwargs["critic_function"] = critic.critic_function()
        kwargs["target_critic_function"] = critic.critic_function(target=True)

        actor_cls = {
            "approximate": ApproximateActor,
            "discrete": DiscreteActor}[actor_type]

        actor = actor_cls.configure(**kwargs) # type: ignore

        agent: Agent = OfflineAgent(
            steps_btw_train=args.steps_btw_train, learn_per_step=args.learn_per_step,
            memory_size=args.memory_size,
            batch_size=args.batch_size, alpha=args.alpha, beta=args.beta,
            actor=actor, critic=critic)
    elif args.algo == "a2c":

        actor = A2CActor.configure(
            action_space=eval_env.action_space,
            observation_space=eval_env.observation_space,
            nb_layers=args.nb_layers, hidden_size=args.hidden_size,
            dt=args.dt, c_entropy=args.c_entropy, normalize=args.normalize_state)

        critic = A2CCritic.configure(
            dt=args.dt, gamma=args.gamma,
            observation_space=eval_env.observation_space,
            nb_layers=args.nb_layers, hidden_size=args.hidden_size,
            noscale=args.noscale, normalize=args.normalize_state)

        agent = A2CAgent(T=args.n_step, actor=actor, critic=critic,
                         opt_name=args.optimizer, lr=args.lr,
                         dt=args.dt, weight_decay=args.weight_decay)
    elif args.algo == "ppo":

        actor = PPOActor.configure(
            action_space=eval_env.action_space,
            observation_space=eval_env.observation_space,
            nb_layers=args.nb_layers, hidden_size=args.hidden_size,
            dt=args.dt, c_entropy=args.c_entropy,
            eps_clamp=args.eps_clamp, c_kl=args.c_kl, normalize=args.normalize_state
        )

        critic = PPOCritic.configure(
            dt=args.dt, gamma=args.gamma,
            observation_space=eval_env.observation_space,
            nb_layers=args.nb_layers, hidden_size=args.hidden_size,
            noscale=args.noscale, eps_clamp=args.eps_clamp,
            normalize=args.normalize_state)

        agent = PPOAgent(
            T=args.n_step, actor=actor, critic=critic,
            learn_per_step=args.learn_per_step,
            batch_size=args.batch_size, opt_name=args.optimizer,
            lr=args.lr, dt=args.dt,
            weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown algorithm {args.algo}")

    return agent, env, eval_env
