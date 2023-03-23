from argparse import Namespace, ArgumentParser
import copy

import numpy as np
from matplotlib import pyplot as plt
import torch

from src.run import run_hes_trial
from alpine import AlpineN1, plot_alpine_2d
import neatplot


neatplot.set_style()
neatplot.update_rc('axes.grid', False)
neatplot.update_rc('font.size', 16)
neatplot.update_rc('text.usetex', False)


parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--num_dim", type=int, default=2)
parser.add_argument("--num_action", type=int, default=2)
parser.add_argument("--algo", type=str, default='kg')
args = parser.parse_args()


# Initialize Alpine function
alpine = AlpineN1(d=args.num_dim, x_scale=0.5, y_scale=0.05)
alpine_bounds = alpine.get_bounds()


def run_script():
    """Run topk with h-entropy-search."""

    # Configure hes trial
    config = Namespace(
        seed = args.seed,
        num_iteration = 100,
        num_initial_points = 10,
        bounds_design = alpine_bounds,
        bounds_action = alpine_bounds,
        num_dim_design = args.num_dim,
        num_dim_action = args.num_dim,
        num_action = args.num_action,
        num_outer_mc = 16,
        num_restarts = 128,
        acq_opt_iter = 200,
        acq_opt_lr = 0.1,
        num_candidates = 1,
        func_is_noisy = False,
        func_noise = 0.1,
        plot_iters = list(range(0, 101, 3)),
    )
    # --- app-specific below
    config.app = 'topk'
    config.fname = 'alpine'
    config.algo = args.algo
    config.dist_weight = 20.0 * alpine.x_scale
    config.dist_threshold = 2.5 * alpine.x_scale
    config.save_dir = f'experiments/opt_{config.app}_{config.fname}_{config.algo}_seed{config.seed}'

    # Run hes trial
    run_hes_trial(func, config, plot_topk_alpine, eval_topk)


def func(x):
    """Alpine function with torch tensor input/output."""
    y = alpine(x, tensor=True)
    return y


@np.vectorize
def func_vec(x, y):
    """Vectorized alpine for contour plot. Return f on input = (x, y)."""
    inp = np.array([x, y]).reshape(-1)
    return alpine(inp)


def plot_function_contour(ax):
    gridwidth = 0.1
    plot_alpine_2d(ax=ax)


def plot_data(ax, data):
    data_x = copy.deepcopy(data.x.detach()).numpy()
    for xi in data_x:
        xi = alpine.transform_to_domain(xi)
        ax.plot(xi[0], xi[1], 'o', color='black', markersize=6)


def plot_next_query(ax, next_x):
    next_x = copy.deepcopy(next_x.detach()).numpy().reshape(-1)
    next_x = alpine.transform_to_domain(next_x)
    ax.plot(next_x[0], next_x[1], 'o', color='deeppink', markersize=7)


def plot_settings(ax, config):
    bounds_plot = alpine.get_bounds(original=True)
    ax.set(xlabel='$x_1$', ylabel='$x_2$', xlim=bounds_plot, ylim=bounds_plot)

    # Set title
    if config.algo == 'hes':
        title = 'H-Entropy Search'
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
        x_actions = alpine.transform_to_domain(x_actions)
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


def plot_optimal_action(ax, optimal_action):
    for x_action in optimal_action:
        x_action = alpine.transform_to_domain(x_action)
        ax.plot(x_action[0], x_action[1], '*', mfc='gold', mec='darkgoldenrod', markersize=40)


def plot_groundtruth_optimal_action(ax, config):

    if config.num_action <= 3:
        centers = [[4.0, 4.0], [2.45, 4.0], [4.0, 2.45]]
    elif config.num_action == 5:
        centers = [[4.0, 4.0], [2.45, 4.0], [4.0, 2.45], [1.0, 4.0], [4.0, 1.0]]
    gt_optimal_action = np.array(centers)

    for x_action in gt_optimal_action:
        x_action = alpine.transform_to_domain(x_action)
        ax.plot(x_action[0], x_action[1], 's', color='blue', markersize=7)


def plot_topk_alpine(config, next_x, data, action_samples, iteration):
    """Plotting for topk."""
    if iteration in config.plot_iters:
        fig, ax = plt.subplots(figsize=(6, 6))

        plot_function_contour(ax)
        plot_action_samples(ax, action_samples, config)
        plot_groundtruth_optimal_action(ax, config)
        plot_data(ax, data)
        plot_next_query(ax, next_x)
        plot_settings(ax, config)

        # Save plot and close
        neatplot.save_figure(f'topk_{iteration}', 'pdf', config.save_dir)
        plt.close()


def plot_eval_topk_alpine(config, next_x, data, optimal_action, iteration):
    """Plotting for topk."""
    if iteration in config.plot_iters:
        fig, ax = plt.subplots(figsize=(6, 6))

        plot_function_contour(ax)
        plot_data(ax, data)
        plot_optimal_action(ax, optimal_action)
        plot_groundtruth_optimal_action(ax, config)
        plot_next_query(ax, next_x)
        plot_settings(ax, config)

        # Save plot and close
        neatplot.save_figure(f'topk_eval_{iteration}', 'pdf', config.save_dir)
        plt.close()


def eval_topk(qhes, config, data, next_x, iteration):
    """Return evaluation metric."""
    # Initialize X_action
    X_design, X_action = qhes.initialize_tensors_topk(data)

    # Set value function
    value_function = qhes.value_function_cls(
        model=qhes.model,
        sampler=qhes.inner_sampler,
        dist_weight=config.dist_weight,
        dist_threshold=config.dist_threshold,
    )

    # Optimize qhes_topk
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
    plot_eval_topk_alpine(config, next_x, data, optimal_action, iteration)

    # Return eval_metric and eval_data (or None)
    return eval_metric, optimal_action


if __name__ == "__main__":
    run_script()
