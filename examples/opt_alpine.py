#!/usr/bin/env python
''''''

__author__      = ''
__copyright__   = 'Copyright 2022, Stanford University'

from argparse import Namespace, ArgumentParser
from matplotlib import pyplot as plt
import os, copy, torch, numpy as np, neatplot
from src.run import run_hes_trial
from alpine import AlpineN1, plot_alpine_2d


neatplot.set_style()
neatplot.update_rc('axes.grid', False)
neatplot.update_rc('font.size', 16)
neatplot.update_rc('text.usetex', False)


parser = ArgumentParser()

# general arguments
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--n_dim", type=int, default=2)
parser.add_argument("--n_actions", type=int, default=1)
parser.add_argument("--algo", type=str, default="hes")
parser.add_argument("--gpuid", type=int, default=0)
parser.add_argument("--truncated_map", choices=("True", "False"), default="True")
parser.add_argument("--lookahead_steps", type=int, default=1)

# MC approximation
parser.add_argument("--n_samples", type=int, default=8)

# optimizer
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--acq_opt_lr", type=float, default=0.1)
parser.add_argument("--acq_opt_iter", type=int, default=200)
parser.add_argument("--n_restarts", type=int, default=128)

# amortization
parser.add_argument("--amortized_vi", choices=("True", "False"), default="True")
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--activation", type=str, default="relu")
parser.add_argument("--hidden_coeff", type=int, default=1)
parser.add_argument("--policy_transfer", choices=("True", "False"), default="False")
parser.add_argument("--joint_init", choices=("True", "False"), default="True")

# baseline
parser.add_argument("--baseline", choices=("True", "False"), default="False")
parser.add_argument("--baseline_n_layers", type=int, default=2)
parser.add_argument("--baseline_hidden_coeff", type=int, default=1)
parser.add_argument("--baseline_activation", type=str, default="relu")
parser.add_argument("--baseline_lr", type=float, default=0.1)

# resampling
parser.add_argument("--n_resampling_max", type=int, default=1)
parser.add_argument("--n_resampling_improvement_threadhold", type=float, default=0.01)
"""When n_resampling_max == 1 and n_resampling_improvement_threadhold is small, we have 
the orange curve. n_resampling_max is large and n_resampling_improvement_threadhold is
large, we have the pink curve (closer to stochastic gradient descent). We can interpolate
between these 2 options by setting both hyperparameters to some moderate value. """

args = parser.parse_args()
args.amortized_vi = True if args.amortized_vi == "True" else False
args.policy_transfer = True if args.policy_transfer == "True" else False
args.baseline = True if args.baseline == "True" else False
args.truncated_map = True if args.truncated_map == "True" else False
args.joint_init = True if args.joint_init == "True" else False
assert args.lookahead_steps > 0

# Initialize Alpine function
alpine = AlpineN1(d=args.n_dim, x_scale=0.5, y_scale=0.05)
alpine_bounds = alpine.get_bounds()

# Set torch device settings
torch_device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")
# torch_device = torch.device("cpu")
torch_dtype = torch.double
torch.set_num_threads(os.cpu_count())

def run_script():
    """Run topk with h-entropy-search."""

    # Configure hes trial
    config = Namespace(
        device = torch_device,
        torch_dtype = torch_dtype,
        seed = args.seed,
        n_iteration = 100,
        n_initial_points = 10,
        bounds_design = alpine_bounds,
        bounds_action = alpine_bounds,
        n_dim_design = args.n_dim,
        n_dim_action = args.n_dim,
        n_actions = args.n_actions,
        n_samples = args.n_samples,
        n_restarts = args.n_restarts,
        acq_opt_iter = args.acq_opt_iter,
        acq_opt_lr = args.acq_opt_lr,
        optimizer = args.optimizer,
        policy_transfer = args.policy_transfer,
        n_resampling_max = args.n_resampling_max,
        n_resampling_improvement_threadhold = args.n_resampling_improvement_threadhold,
        joint_init = args.joint_init,
        baseline = args.baseline,
        n_candidates = 1,
        func_is_noisy = False,
        func_noise = 0.1,
        plot_iters = list(range(0, 101, 1)),
    )

    # --- nn-specific below
    config.n_layers = args.n_layers
    config.activation = args.activation
    config.hidden_coeff = args.hidden_coeff
    config.truncated_map = args.truncated_map
    
    # -- non-myopic-specific below
    config.amortized_vi = args.amortized_vi
    config.lookahead_steps = args.lookahead_steps

    # --- app-specific below
    config.app = 'topk'
    config.fname = 'alpine'
    config.algo = args.algo
    config.dist_weight = 20.0 * alpine.x_scale
    config.dist_threshold = 2.5 * alpine.x_scale
    config.save_dir = f'experiments/opt_{config.app}_{config.fname}_{config.algo}_seed{config.seed}_n_samples{config.n_samples}_n_restarts{config.n_restarts}_amortized_vi{config.amortized_vi}_lookahead_steps{config.lookahead_steps}_n_layers{config.n_layers}_activation{config.activation}_hidden_coeff{config.hidden_coeff}_acq_opt_lr{config.acq_opt_lr}_acq_opt_iter{config.acq_opt_iter}_n_resampling_max{config.n_resampling_max}_baseline{config.baseline}'

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
    data_x = copy.deepcopy(data.x.cpu().detach()).numpy()
    for xi in data_x:
        xi = alpine.transform_to_domain(xi)
        ax.plot(xi[0], xi[1], 'o', color='black', markersize=6)


def plot_next_query(ax, next_x):
    next_x = copy.deepcopy(next_x.cpu().detach()).numpy().reshape(-1)
    next_x = alpine.transform_to_domain(next_x)
    ax.plot(next_x[0], next_x[1], 'o', color='deeppink', markersize=7)


def plot_settings(ax, config):
    bounds_plot = alpine.get_bounds(original=True)
    bounds_plot_ext = [bounds_plot[0] - 0.5, bounds_plot[1] + 0.5]
    ax.set(xlabel='$x_1$', ylabel='$x_2$', xlim=bounds_plot_ext, ylim=bounds_plot_ext)

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
    action_samples = copy.deepcopy(action_samples.cpu().detach()).numpy()
    for x_actions in action_samples:
        x_actions = alpine.transform_to_domain(x_actions)
        lines2d = ax.plot(x_actions[0][0], x_actions[0][1], '*', mec='k', markersize=15)
        if config.n_actions >= 2:
            color = lines2d[0].get_color()
            ax.plot(x_actions[1][0], x_actions[1][1], '*', mfc=color, mec='k', markersize=15)
            line_1_x = [x_actions[0][0], x_actions[1][0]]
            line_1_y = [x_actions[0][1], x_actions[1][1]]
            ax.plot(line_1_x, line_1_y, '--', color=color)
        if config.n_actions >= 3:
            ax.plot(x_actions[2][0], x_actions[2][1], '*', mfc=color, mec='k', markersize=15)
            line_2_x = [x_actions[1][0], x_actions[2][0]]
            line_2_y = [x_actions[1][1], x_actions[2][1]]
            ax.plot(line_2_x, line_2_y, '--', color=color)


def plot_optimal_action(ax, optimal_action, config):
    for x_action in optimal_action:
        x_action = x_action.squeeze()
        x_action = alpine.transform_to_domain(x_action)
        ax.plot(x_action[0], x_action[1], '*', mfc='gold', mec='darkgoldenrod', markersize=40)


def plot_groundtruth_optimal_action(ax, config):

    if config.n_actions <= 3:
        centers = [[4.0, 4.0], [2.45, 4.0], [4.0, 2.45]]
    elif config.n_actions == 5:
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
        plot_optimal_action(ax, optimal_action, config)
        plot_data(ax, data)
        plot_groundtruth_optimal_action(ax, config)
        plot_next_query(ax, next_x)
        plot_settings(ax, config)

        # Save plot and close
        neatplot.save_figure(f'topk_eval_{iteration}', 'pdf', config.save_dir)
        plt.close()


def eval_topk(batch_hes, config, data, next_x, iteration):
    """Return evaluation metric."""
    # Initialize X_action
    X_design, X_action = batch_hes.initialize_tensors_topk(data)

    # Set value function
    batch_hes_topk = batch_hes.batch_hes(
        model=batch_hes.model,
        config=config,
        sampler=batch_hes.inner_sampler,
        dist_weight=config.dist_weight,
        dist_threshold=config.dist_threshold,
    )

    # Optimize hes_topk
    learnable_params = [X_design]
    parameters = []
    assert config.lookahead_steps <= 3
    if config.amortized_vi == True:
        if config.lookahead_steps >= 1:
            parameters += [batch_hes.batch_mlp_a[i].parameters() for i in range(config.n_restarts)]
        if config.lookahead_steps >= 2:
            parameters += [batch_hes.batch_mlp_x1[i].parameters() for i in range(config.n_restarts)]
        if config.lookahead_steps >= 3:
            parameters += [batch_hes.batch_mlp_x2[i].parameters() for i in range(config.n_restarts)]
        for i in range(len(parameters)):
            learnable_params += list(parameters[i])
    else:
        if config.lookahead_steps >= 1: parameters += X_action
        if config.lookahead_steps >= 2: parameters += batch_hes.batch_x1s
        if config.lookahead_steps >= 3: parameters += batch_hes.batch_x2s
    optimizer = torch.optim.Adam(learnable_params, lr=config.acq_opt_lr)

    for i in range(config.acq_opt_iter):
        optimizer.zero_grad()
        optimization_objective = -1 * batch_hes_topk(X_action[:2, :2, ...]).sum()
        optimization_objective.backward()
        optimizer.step()
        X_design.data.clamp_(config.bounds_design[0], config.bounds_design[1])
        X_action.data.clamp_(config.bounds_action[0], config.bounds_action[1])

        if (i+1) % (config.acq_opt_iter//5) == 0 or i == config.acq_opt_iter-1:
            print('Eval:', i+1, optimization_objective.item())

    X_action = X_action[:2, :2, ...]
    acq_values = batch_hes_topk(X_action)

    optimal_action = X_action[0][0].cpu().detach().numpy()
    eval_metric = acq_values[0].cpu().detach().numpy().tolist()

    print(f'Eval optimal_action: {optimal_action}')

    # Plot optimal_action in special eval plot here
    plot_eval_topk_alpine(config, next_x, data, optimal_action, iteration)

    # Return eval_metric and eval_data (or None)
    return eval_metric, optimal_action


if __name__ == "__main__":
    run_script()
