import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(11, 6))
plt.style.use('ggplot')
ax = plt.axes()
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.margins(9, 0)
MIN_DT = 5e-4
MAX_DT = 5e-2
bot_margin = 1
left_margin = 2
width = 5
font_size = 35

color_range = np.linspace(np.log(MIN_DT), np.log(MAX_DT), 50)
cnorm = matplotlib.colors.Normalize(vmin=np.log(MIN_DT) - 3, vmax=np.log(MAX_DT) + .1)
cm1 = matplotlib.cm.ScalarMappable(norm=cnorm, cmap='Blues')
cm2 = matplotlib.cm.ScalarMappable(norm=cnorm, cmap='Reds')
line_color_blue = np.array([cm1.to_rgba(c) for c in color_range])[np.newaxis, ...]
line_color_blue = np.tile(line_color_blue, (width, 1, 1))
line_color_white = np.array([[1., 1., 1., 1.] for c in color_range])[np.newaxis, ...]
line_color_white = np.tile(line_color_white, (width, 1, 1))
line_color_red = np.array([cm2.to_rgba(c) for c in color_range])[np.newaxis, ...]
line_color_red = np.tile(line_color_red, (width, 1, 1))
line_color = np.concatenate([line_color_blue, line_color_white, line_color_red], axis=0)

ax.text(0, 3 * width + bot_margin, f"{MIN_DT:3.2e}", color="dimgrey", verticalalignment='top', horizontalalignment='center', size=int(font_size * 2 / 3))
ax.text(50, 3 * width + bot_margin, f"{MAX_DT:3.2e}", color="dimgrey", verticalalignment='top', horizontalalignment='center', size=int(font_size * 2 / 3))
ax.text(0, width + bot_margin, f"{MIN_DT:3.2e}", color="dimgrey", verticalalignment='top', horizontalalignment='center', size=int(font_size * 2 / 3))
ax.text(50, width + bot_margin, f"{MAX_DT:3.2e}", color="dimgrey", verticalalignment='top', horizontalalignment='center', size=int(font_size * 2 / 3))
ax.text(25, 3 * width + 3 * bot_margin, "Time discretization (log scale)", verticalalignment='top', horizontalalignment='center', size=int(font_size * 3 / 4), color="dimgrey")
ax.text(-left_margin, width / 2, "DDPG", color="dimgrey", verticalalignment='center', horizontalalignment='right', size=font_size)
ax.text(-left_margin, 2 * width + width / 2, "AU", color="dimgrey", verticalalignment='center', horizontalalignment='right', size=font_size)

plt.imshow(line_color)
plt.subplots_adjust(left=.2, right=.9)
plt.savefig('legend.png')
