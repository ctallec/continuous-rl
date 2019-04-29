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


### Run examples

Bipedal walker with DDPG:
```bash
python main.py --algo ddpg --steps_btw_train 10 --noise_type coherent --batch_size 256 --hidden_size 256 --nb_layers 1 --gamma 0.8 --nb_steps 100 --sigma 1.5 --theta 7.5 --nb_train_env 256 --nb_eval_env 64 --memory_size 1000000 --learn_per_step 50 --eval_gap 0.05   --weight_decay 0.0   --tau 0.9 --optimizer rmsprop --env_id bipedal_walker --time_limit 10 --dt 0.02 --normalize_state  --lr 0.1 --policy_lr 0.02  --noscale --nb_true_epochs 20 --logdir ~/logdir
```


Bipedal walker with DAU:
```bash
python main.py --algo cdau --steps_btw_train 10 --noise_type coherent --batch_size 256 --hidden_size 256 --nb_layers 1 --gamma 0.8 --nb_steps 100 --sigma 1.5 --theta 7.5 --nb_train_env 256 --nb_eval_env 64 --memory_size 1000000 --learn_per_step 50 --eval_gap 0.05   --weight_decay 0.0   --tau 0.0 --optimizer rmsprop --env_id bipedal_walker --time_limit 10 --dt 0.02 --normalize_state  --lr 0.1 --policy_lr 0.02   --nb_true_epochs 20 --logdir ~/logdir
```

Bipedal walker with A2C:
```bash
python main.py --algo a2c --steps_btw_train 20 --noise_type coherent --batch_size 256 --hidden_size 256 --nb_layers 1 --gamma 0.8 --nb_steps 100 --sigma 1.5 --theta 7.5 --nb_train_env 256 --nb_eval_env 64 --memory_size 1000000 --learn_per_step 50 --eval_gap 0.05   --weight_decay 0.0   --tau 0.9 --optimizer rmsprop --env_id bipedal_walker --time_limit 10 --dt 0.02 --normalize_state  --lr 0.01 --policy_lr 0.001 --c_entropy 0.0001 --n_step 20 --nb_true_epochs 20 --logdir ~/logdir
```