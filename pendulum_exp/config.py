from functools import partial
from policies.offline_policy import OfflinePolicy
from abstract import Env
from actors import DiscreteActor, SampledActor, ApproximateActor
from critics import AdvantageCritic
from envs.utils import make_env
from envs.vecenv import SubprocVecEnv
from noise import setup_noise

def configure(args):
    noise = setup_noise(
        noise_type=args.noise_type, sigma=args.sigma,
        theta=args.theta, dt=args.dt, sigma_decay=lambda _: 1.)

    env_fn = partial(make_env, env_id=args.env_id,
                     dt=args.dt, time_limit=args.time_limit)

    env: Env = SubprocVecEnv([env_fn() for _ in range(args.nb_train_env)])
    eval_env: Env = SubprocVecEnv([env_fn() for _ in range(args.nb_eval_env)])

    actor_type, critic_type = args.algo.split('_')

    kwargs = dict(
        dt=args.dt, gamma=args.gamma, lr=args.lr, optimizer=args.optimizer,
        action_space=env.action_space, observation_space=env.observation_space,
        nb_layers=args.nb_layers, hidden_size=args.hidden_size,
        normalize=args.normalize_state, weight_decay=args.weight_decay, noise=noise
    )

    critic_cls = {
        "advantage": AdvantageCritic
    }[critic_type]

    critic = critic_cls.configure(**kwargs)
    kwargs["critic_function"] = critic.critic_function

    actor_cls = {
        "sampled": SampledActor,
        "approximate": ApproximateActor,
        "discrete": DiscreteActor}[actor_type]

    actor = actor_cls.configure(**kwargs)

    policy = OfflinePolicy(
        steps_btw_train=args.steps_btw_train, learn_per_step=args.learn_per_step,
        memory_size=args.memory_size,
        batch_size=args.batch_size, alpha=args.alpha, beta=args.beta,
        actor=actor, critic=critic)

    return policy, env, eval_env
