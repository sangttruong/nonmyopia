"""
Simple example with synthetic top-k estimation setup.
"""

from argparse import Namespace, ArgumentParser
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import cm


# Parse arguments
parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--n_init", type=int, default=10)
parser.add_argument("--n_iter", type=int, default=100)
parser.add_argument("--acq_str", type=str, default="hes")
args = parser.parse_args()

# Set seeds
print(f"*[INFO] Seed: {args.seed}")
np.random.seed(args.seed)

# Set synthetic function
f_0 = lambda x: 2 * np.abs(x) * np.sin(x)
f = lambda x_list: np.sum([f_0(x) for x in x_list])

# Set algorithm  details
n_dim = 2
domain = [[-10, 10]] * n_dim
len_path = 150
k = 10
x_set = uniform_random_sample_domain(domain, len_path)


# Set vectorized function (for contour plot)
@np.vectorize
def f_vec(x, y):
    """Return f on input = (x, y)."""
    return f((x, y))


# Plot
fig, ax = plt.subplots(figsize=(6, 6))
# --- plot function contour
grid = 0.1
xpts = np.arange(domain[0][0], domain[0][1], grid)
ypts = np.arange(domain[1][0], domain[1][1], grid)
X, Y = np.meshgrid(xpts, ypts)
Z = f_vec(X, Y)
ax.contour(X, Y, Z, 20, cmap=cm.Greens_r, zorder=0)
# --- plot x_set
x_set_arr = np.array(x_set)
ax.plot(x_set_arr[:, 0], x_set_arr[:, 1], '.', color='#C0C0C0', markersize=8)
plt.show()
