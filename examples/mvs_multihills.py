from argparse import Namespace, ArgumentParser
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
neatplot.update_rc('font.size', 16)


parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--num_dim", type=int, default=2)
parser.add_argument("--algo", type=str, default='hes')
args = parser.parse_args()


# Initialize Multihills
centers = [[0.2, 0.2], [0.2, 0.8], [0.8, 0.8]]
widths = [0.2, 0.2, 0.2]
weights = [0.4, 0.7, 1.0]
multihills = Multihills(centers, widths, weights)



def run_script():
    """Run mvs with h-entropy-search."""

    # Configure hes trial
    config = Namespace(
        seed = args.seed,
        num_iteration = 90,
        num_initial_points = 10,
        bounds_design = multihills_bounds,
        bounds_action = multihills_bounds,
        num_dim_design = args.num_dim,
        num_dim_action = args.num_dim,
        num_outer_mc = 16,
        #num_restarts = 128,
        num_restarts = 4, #### NOTE
        acq_opt_iter = 200,
        acq_opt_lr = 0.1,
        num_candidates = 1,
        func_is_noisy = False,
        func_noise = 0.1,
        plot_iters = [1, 5, 25, 50, 100],
    )
    # --- app-specific below
    config.app = 'mvs'
    config.fname = 'multihills'
    config.algo = args.algo
    config.val_tuple = (
        4.0,    # (0.8, 0.8)
        3.0,
        2.0,
        1.0,
        0.05,
    )
    config.num_action = len(config.val_tuple)
    config.save_dir = f'experiments/{config.app}_{config.fname}_{config.algo}_seed{config.seed}'

    # Run hes trial
    run_hes_trial(func, config, plot_mvs, eval_mvs)


def func(x):
    """Function with torch tensor input/output."""
    x_list = [xi.detach().numpy().tolist() for xi in x]
    y_list = multihills(x_list)
    y_tensor = torch.tensor(np.array(y_list).reshape(-1, 1))
    return y_tensor


@np.vectorize
def func_vec(x, y):
    """Vectorized function for contour plot. Return f on input = (x, y)."""
    return multihills((x, y))


def plot_function_contour(ax):
    gridwidth = 0.01
    xpts = np.arange(multihills_bounds[0], multihills_bounds[1], gridwidth)
    ypts = np.arange(multihills_bounds[0], multihills_bounds[1], gridwidth)
    X, Y = np.meshgrid(xpts, ypts)
    Z = func_vec(X, Y)
    ax.contour(X, Y, Z, 50, cmap=cm.Greens_r, zorder=0)


def plot_data(ax, data):
    data_x = copy.deepcopy(data.x.detach()).numpy()
    for xi in data_x:
        ax.plot(xi[0], xi[1], 'o', color='black', markersize=6)


def plot_next_query(ax, next_x):
    next_x = copy.deepcopy(next_x.detach()).numpy().reshape(-1)
    next_x = next_x.reshape(-1).tolist()
    ax.plot(next_x[0], next_x[1], 'o', color='deeppink', markersize=7)


def plot_settings(ax, config):
    ax.set(xlabel='$x_1$', ylabel='$x_2$', xlim=multihills_bounds, ylim=multihills_bounds)

    # Set title
    if config.algo == 'hes':
        title = '$H_{\ell, \mathcal{A}}$-Entropy Search'
    elif config.algo == 'kg':
        title = 'Knowledge Gradient'
    if config.algo == 'rs':
        title = 'Random Search'
    if config.algo == 'us':
        title = 'Uncertainty Sampling'
    #ax.set(title=title)
    ax.set_title(label=title, fontdict={'fontsize': 25})


def plot_action_samples(ax, action_samples, config):
    action_samples = copy.deepcopy(action_samples.detach()).numpy()

    for x_actions in action_samples:
        lines2d = ax.plot(x_actions[0][0], x_actions[0][1], '*', mec='k', markersize=15)
        color = lines2d[0].get_color()

        for idx in range(1, len(x_actions)):
            line_x = (x_actions[idx-1][0], x_actions[idx][0])
            line_y = (x_actions[idx-1][1], x_actions[idx][1])
            ax.plot(line_x, line_y, '--', color=color)
            ax.plot(x_actions[idx][0], x_actions[idx][1], '*', mfc=color, mec='k', markersize=15)


def plot_optimal_action(ax, optimal_action):
    for x_action in optimal_action:
        ax.plot(x_action[0], x_action[1], 'D', mfc='red', mec='darkred', markersize=20)

    for idx in range(1, len(optimal_action)):
        xa1 = optimal_action[idx-1]
        xa2 = optimal_action[idx]
        ax.plot([xa1[0], xa2[0]], [xa1[1], xa2[1]], '--', color='darkred')


def plot_groundtruth_optimal_action(ax, config):
    pass


def plot_mvs(config, next_x, data, action_samples, iteration):
    """Plotting for mvs."""
    if iteration in config.plot_iters:
        fig, ax = plt.subplots(figsize=(6, 6))

        plot_function_contour(ax)
        plot_action_samples(ax, action_samples, config)
        plot_groundtruth_optimal_action(ax, config)
        plot_data(ax, data)
        plot_next_query(ax, next_x)
        plot_settings(ax, config)

        # Save plot and close
        neatplot.save_figure(f'mvs_{iteration}', 'pdf', config.save_dir)
        plt.close()


def plot_eval_mvs(config, next_x, data, optimal_action, iteration):
    """Plotting for mvs."""
    if iteration in config.plot_iters:
        fig, ax = plt.subplots(figsize=(6, 6))

        plot_function_contour(ax)
        plot_data(ax, data)
        plot_optimal_action(ax, optimal_action)
        plot_groundtruth_optimal_action(ax, config)
        plot_next_query(ax, next_x)
        plot_settings(ax, config)

        # Save plot and close
        neatplot.save_figure(f'mvs_eval_{iteration}', 'pdf', config.save_dir)
        plt.close()


def eval_mvs(qhes, config, data, next_x, iteration):
    """Return evaluation metric."""
    # Initialize X_action
    X_design, X_action = qhes.initialize_tensors_mvs(data)
    #X_design, X_action = qhes.initialize_design_action_tensors(data)

    # Set value function
    value_function = qhes.value_function_cls(
        model=qhes.model,
        val_tuple=config.val_tuple,
        sampler=qhes.inner_sampler,
    )

    # Optimize qhes mvs
    optimizer = torch.optim.Adam([X_design, X_action], lr=config.acq_opt_lr)
    for i in range(config.acq_opt_iter):
        optimizer.zero_grad()
        losses = -value_function(X_action[:2, :2, :, :]).mean(dim=0)
        loss = losses.sum()
        loss.backward(retain_graph=True)
        optimizer.step()
        X_design.data.clamp_(config.bounds_design[0], config.bounds_design[1])
        X_action.data.clamp_(config.bounds_action[0], config.bounds_action[1])

        if (i+1) % (config.acq_opt_iter//5) == 0 or i == config.acq_opt_iter-1:
            print('Eval:', i+1, loss.item())

    X_action = X_action[:2, :2, :, :]
    acq_values = value_function(X_action)

    optimal_action = X_action[0][0].detach().numpy()
    eval_metric = acq_values[0][0].detach().numpy().tolist()
    print(f'Eval optimal_action: {optimal_action}')

    # Plot optimal_action in special eval plot here
    plot_eval_mvs(config, next_x, data, optimal_action, iteration)

    # Return eval_metric and eval_data (or None)
    return eval_metric, optimal_action


if __name__ == "__main__":
    run_script()
