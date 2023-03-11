import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from src.utilities import uniform_random_sample_domain

from branin import branin, branin_normalized


domain = [(-5, 10), (0, 15)]


n_samp = 100000
x_list = uniform_random_sample_domain(domain, n_samp)
y_list = branin(x_list)

print(f'Minimum y = {np.min(y_list)}')
print(f'Maximum y = {np.max(y_list)}')


x_list = [(-5, 0), (-5, 15), (10, 0), (10, 15), (0, 0)] 
y_list = branin(x_list)

x_list_norm = [(0, 0), (0, 1), (1, 0), (1, 1), (0.5, 0.5)] 
y_list_norm = branin_normalized(x_list_norm)

ylims = (0.3978, 300.0)
y_list_norm_reverted = y_list_norm * (ylims[1] - ylims[0]) + ylims[0]

print(f'x_list = {x_list}')
print(f'y_list = {y_list}')
print(f'x_list_norm = {x_list_norm}')
print(f'y_list_norm = {y_list_norm}')
print(f'y_list_norm_reverted = {y_list_norm_reverted}')




# Plot
@np.vectorize
def f_vec(x, y):
 """Return f on input = (x, y)."""
 return branin((x, y))


fig, ax = plt.subplots(figsize=(6, 6))
# --- plot function contour
gridwidth = 0.1
xpts = np.arange(domain[0][0], domain[0][1], gridwidth)
ypts = np.arange(domain[1][0], domain[1][1], gridwidth)
X, Y = np.meshgrid(xpts, ypts)
Z = f_vec(X, Y)
#ax.contour(X, Y, Z, 20, cmap=cm.Greens_r, zorder=0)
ax.contour(X, Y, Z, 50, cmap=cm.Greens_r, zorder=0)
# --- plot best_design
ax.plot([0, 1], [2, 3], '.', color='r', markersize=10)
plt.show()
