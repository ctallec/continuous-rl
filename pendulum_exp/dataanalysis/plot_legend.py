import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
ax = plt.axes()
ax.grid(False)
ax.set_xticks([])
ax.set_xlabel("Time discretization (log scale)")
ax.set_yticks([])
ax.margins(9, 0)
MIN_DT = 5e-4
MAX_DT = 5e-2
bot_margin = .2
width = 5

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

ax.text(0, 3 * width + bot_margin, f"{MIN_DT:3.2e}", color="dimgrey", verticalalignment='top', horizontalalignment='center')
ax.text(50, 3 * width + bot_margin, f"{MAX_DT:3.2e}", color="dimgrey", verticalalignment='top', horizontalalignment='center')
ax.text(-1, width / 2, "DDPG", color="dimgrey", verticalalignment='center', horizontalalignment='right', size=14)
ax.text(-1, 2 * width + width / 2, "Advantage updating", color="dimgrey", verticalalignment='center', horizontalalignment='right', size=13)
print(np.exp(color_range))

plt.imshow(line_color)
plt.show()
input()
