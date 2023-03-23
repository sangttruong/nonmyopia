from argparse import Namespace
import copy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import torch

from src.run import run_hes_trial
from multihills import Multihills, multihills_bounds
import neatplot


neatplot.set_style()
neatplot.update_rc('axes.grid', False)


# Initialize Multihills
centers = [[0.2, 0.2], [0.2, 0.8], [0.8, 0.8]]
widths = [0.2, 0.2, 0.2]
weights = [1.0, 0.5, 1.0]
multihills = Multihills(centers, widths, weights)


def run_script():
    """Run topk with h-entropy-search."""

    # Configure hes trial
    config = Namespace(
        seed = 11,
        num_iteration = 40,
        num_initial_points = 5,
        bounds_design = multihills_bounds,
        bounds_action = multihills_bounds,
        num_dim_design = 2,
        num_dim_action = 2,
        num_action = 2,
        num_outer_mc = 16,
        num_restarts = 128,
        acq_opt_iter = 100,
        acq_opt_lr = 0.1,
        num_candidates = 1,
        func_is_noisy = False,
        func_noise = 0.1,
    )
    # --- app-specific below
    config.app = 'topk'
    config.algo = 'hes'
    config.fname = 'multihills'
    config.dist_weight = 20.0
    config.dist_threshold = 0.5
    config.save_dir = f'experiments/{config.app}_{config.fname}_{config.algo}_seed{config.seed}'

    # Run hes trial
    run_hes_trial(func, config, plot_topk_multihills)


def func(x):
    """Multihills function with torch tensor input/output."""
    x_list = [xi.detach().numpy().tolist() for xi in x]
    y_list = multihills(x_list)
    y_tensor = torch.tensor(np.array(y_list).reshape(-1, 1))
    return y_tensor


@np.vectorize
def func_vec(x, y):
    """Vectorized multihills for contour plot. Return f on input = (x, y)."""
    return multihills((x, y))


def plot_topk_multihills(config, next_x, data, action_samples, iteration):
    """Plotting for topk."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # --- plot function contour
    gridwidth = 0.01
    xpts = np.arange(multihills_bounds[0], multihills_bounds[1], gridwidth)
    ypts = np.arange(multihills_bounds[0], multihills_bounds[1], gridwidth)
    X, Y = np.meshgrid(xpts, ypts)
    Z = func_vec(X, Y)
    ax.contour(X, Y, Z, 50, cmap=cm.Greens_r, zorder=0)

    # --- plot next query
    next_x = copy.deepcopy(next_x.detach()).numpy()
    next_x = next_x.reshape(-1).tolist()
    ax.plot(next_x[0], next_x[1], 'o', color='deeppink', markersize=12)

    # --- plot data so far
    data_x = copy.deepcopy(data.x.detach()).numpy()
    for xi in data_x:
        ax.plot(xi[0], xi[1], 'o', color='grey', markersize=8)

    # --- plot optimal actions
    action_samples = copy.deepcopy(action_samples.detach()).numpy()
    for x_actions in action_samples.tolist():
        lines2d = ax.plot(x_actions[0][0], x_actions[0][1], '*', mec='k', markersize=15)
        if config.num_action >= 2:
            color = lines2d[0].get_color()
            ax.plot(x_actions[1][0], x_actions[1][1], '*', mfc=color, mec='k', markersize=15)
            line_1_x = [x_actions[0][0], x_actions[1][0]]
            line_1_y = [x_actions[0][1], x_actions[1][1]]
            ax.plot(line_1_x, line_1_y, '--', color=color)
        if config.num_action >= 3:
            ax.plot(x_actions[2][0], x_actions[2][1], '*', mfc=color, mec='k', markersize=15)
            line_2_x = [x_actions[1][0], x_actions[2][0]]
            line_2_y = [x_actions[1][1], x_actions[2][1]]
            ax.plot(line_2_x, line_2_y, '--', color=color)

    # --- Set plot settings
    ax.set(
        xlabel='$x_1$', ylabel='$x_2$', xlim=multihills_bounds, ylim=multihills_bounds
    )

    # --- save plot and close
    neatplot.save_figure(f'topk_{iteration}', 'png', config.save_dir)
    plt.close()


if __name__ == "__main__":
    run_script()
