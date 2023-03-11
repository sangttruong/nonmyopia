from argparse import Namespace

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import torch
import os
from src.run import run_hes_trial

from branin import (
    branin,
    branin_normalized,
    transform_to_branin_domain,
    transform_to_branin_domain_actions,
)
from branin import domain as branin_domain
import neatplot


neatplot.set_style()
neatplot.update_rc('axes.grid', False)


def run_script():
    """Run topk with h-entropy-search."""

    # Configure hes trial
    config = Namespace(
        seed = 11,
        num_iteration = 30,
        num_initial_points = 5,
        bounds_design = [0.0, 1.0],
        bounds_action = [0.0, 1.0],
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
    config.fname = 'branin'
    config.dist_weight = 10.0
    config.dist_threshold = 0.3
    config.save_dir = f'experiments/{config.app}_{config.fname}_{config.algo}_seed{config.seed}'

    # Run hes trial
    run_hes_trial(func, config, plot_topk_branin)


def func(x):
    """Normalized branin (maximization) function with torch tensor input/output."""
    x_list = [xi.detach().numpy().tolist() for xi in x]
    y_list = -1 * branin_normalized(x_list)
    y_tensor = torch.tensor(np.array(y_list).reshape(-1, 1))
    return y_tensor


@np.vectorize
def func_vec(x, y):
    """Vectorized branin for contour plot. Return f on input = (x, y)."""
    return branin((x, y))


def plot_topk_branin(config, next_x, data, action_samples, iteration):
    """Plotting for topk."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # --- plot function contour
    gridwidth = 0.1
    xpts = np.arange(branin_domain[0][0], branin_domain[0][1], gridwidth)
    ypts = np.arange(branin_domain[1][0], branin_domain[1][1], gridwidth)
    X, Y = np.meshgrid(xpts, ypts)
    Z = func_vec(X, Y)
    ax.contour(X, Y, Z, 50, cmap=cm.Greens_r, zorder=0)

    # --- plot next query
    next_x = transform_to_branin_domain(next_x).reshape(-1).tolist()
    ax.plot(next_x[0], next_x[1], 'o', color='deeppink', markersize=12)

    # --- plot data so far
    data_x = transform_to_branin_domain(data.x)
    for xi in data_x:
        ax.plot(xi[0], xi[1], 'o', color='grey', markersize=8)

    # --- plot optimal actions
    action_samples = transform_to_branin_domain_actions(action_samples)
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
    ax.set(xlabel='$x_1$', ylabel='$x_2$', xlim=[-5.3, 10.3], ylim=[-0.3, 15.3])

    # --- save plot and close
    neatplot.save_figure(f'topk_{iteration}', 'png', config.save_dir)
    plt.close()


if __name__ == "__main__":
    run_script()
