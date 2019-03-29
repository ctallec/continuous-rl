"""Retrieves pendulum figure from the last version of pendulum_final exp."""
import pickle
from typing import List, Dict, Tuple
import os
from os.path import join, isdir
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

timesteps = [1, 2, 3, 4, 5]
nb_ts = len(timesteps)
exp_dir = 'local_logs/pendulum_final_noscale/'
max_time = datetime.strptime("1999_01_01_01_01_01", "%Y_%m_%d_%H_%M_%S")
max_dir = None

for sub_dir in os.listdir(exp_dir):
    time = datetime.strptime(sub_dir, "%Y_%m_%d_%H_%M_%S")
    max_dir = sub_dir if time > max_time or not max_dir else max_dir

assert max_dir
exp_dir = join(exp_dir, max_dir)

exp_dirs: List[str] = []
for sub_dir in os.listdir(exp_dir):
    full_dir = join(exp_dir, sub_dir)
    if isdir(full_dir):
        exp_dirs.append(full_dir)

dt_dict: Dict[Tuple[str, float], str] = {}
max_dt = 0.
nb_dt = 0
for sub_dir in exp_dirs:
    args = pickle.load(open(join(sub_dir, 'args'), 'rb'))
    key = (args.algo, args.dt)
    max_dt = max(args.dt, max_dt)
    if key not in dt_dict:
        dt_dict[key] = sub_dir
        nb_dt += 1
    else:
        dt_dict[key] = sub_dir
assert nb_dt % 2 == 0
nb_dt = nb_dt // 2

val_img_dict: Dict[Tuple[str, float], List[str]] = {k: [] for k in dt_dict}
act_img_dict: Dict[Tuple[str, float], List[str]] = {k: [] for k in dt_dict}
for key, value in dt_dict.items():
    for ts in timesteps:
        val_img_dict[key].append(join(value, 'imgs', f'val_{int(ts * max_dt / key[1])}.png'))
        act_img_dict[key].append(join(value, 'imgs', f'act_{int(ts * max_dt / key[1])}.png'))

dts = sorted(list(set([k[1] for k in dt_dict])))
dts = dts[:-1]
print(f"List of dts: {', '.join([str(dt) for dt in dts])}")
nb_dt = len(dts)

# draw figure
val_ddpg = plt.figure(0)
val_au = plt.figure(1)
act_ddpg = plt.figure(2)
act_au = plt.figure(3)
ddpg_lbl = 'approximate_value'
au_lbl = 'approximate_advantage'
x_plot = 0
y_plot = 0
for y_plot, dt in enumerate(dts):
    for nb_fig, lbl, dic in [
            (0, ddpg_lbl, val_img_dict),
            # (1, au_lbl, val_img_dict),
            (2, ddpg_lbl, act_img_dict)]:
            # (3, au_lbl, act_img_dict)]:
        plt.figure(nb_fig)
        for x_plot, v in enumerate(dic[(lbl, dt)]):
            plt.subplot(nb_dt, nb_ts, x_plot + (nb_dt - y_plot - 1) * nb_ts + 1)
            print(f"At index {x_plot + (nb_dt - y_plot - 1) * nb_ts + 1}, plotting {v}")
            img = mpimg.imread(v)
            plt.axis('off')
            plt.imshow(img, cmap='plasma')

plt.figure(0)
plt.savefig('val_ddpg.png')
# plt.figure(1)
# plt.savefig('val_au.png')
plt.figure(2)
plt.savefig('act_ddpg.png')
# plt.figure(3)
# plt.savefig('act_au.png')
