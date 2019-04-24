# Open source code to reproduce Making deep Q-learning approaches robust to time discretization #
This repository provides code to reproduce the results of **Making deep Q-learning approaches robust to time discretization** [Arxiv](https://arxiv.org/abs/1901.09732), [Blog post](https://ctallec.github.io/continuous-rl/).
In addition, we also provide implementations for some standard reinforcement learning algorithms.

Typical run commands are run from the code subdirectory and take the form
```
python run.py --logdir existing_dir_where_you_want_to_store_your_logs [options]
```
The most critical options are
- `--algo`: which algo to use. Currently, the algorithms available are **ddpg**, **dqn**, **ddau** (for discrete deep advantage updating), **cdau** (for continuous deep advantage updating), **a2c** and **ppo**.
- `--dt`: inverse of the framerate, or discretization timestep. Expect learning times to scale as 1/dt.
- `--noise_type`: wether you want to use temporally coherent noise or independent noise.
- `--gamma`: the discount factor **IN PHYSICAL TIME**. The actual discount factor will be **gamma^dt**.
- `--env_id`: environment to train upon.

Other parameters are algorithm dependent, run `python main.py --help` to get more details, check the code for precise details.
