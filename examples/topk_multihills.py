#!/usr/bin/env python
''' '''

__author__      = ''
__copyright__   = 'Copyright 2022, Stanford University'

from argparse import Namespace, ArgumentParser
import os, copy, torch, numpy as np, neatplot

from matplotlib import pyplot as plt
from matplotlib import cm
from src.run import run_hes_trial
from multihills import Multihills, multihills_bounds


neatplot.set_style()
neatplot.update_rc('axes.grid', False)
neatplot.update_rc('font.size', 16)
neatplot.update_rc('text.usetex', False)


parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--n_dim", type=int, default=2)
parser.add_argument("--n_actions", type=int, default=1)
parser.add_argument("--algo", type=str, default="hes")
parser.add_argument("--gpuid", type=int, default=0)
parser.add_argument("--n_ys", type=int, default=16)
parser.add_argument("--n_restarts", type=int, default=128)
parser.add_argument("--acq_opt_lr", type=float, default=0.1)
parser.add_argument("--acq_opt_iter", type=int, default=100)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--n_resampling", type=int, default=1)

# neural network args
parser.add_argument("--n_layers", type=int, default=3)
parser.add_argument("--activation", type=str, default="relu")
parser.add_argument("--hidden_coeff", type=int, default=2)
parser.add_argument("--policy_transfer", choices=("True", "False"), default="True")

# non-myopic args
parser.add_argument("--amortized_vi", choices=("True", "False"), default="True")
parser.add_argument("--lookahead_steps", type=int, default=1)

args = parser.parse_args()
args.amortized_vi = True if args.amortized_vi == "True" else False
args.policy_transfer = True if args.policy_transfer == "True" else False

# Initialize Multihills
centers = [[0.2, 0.2], [0.2, 0.8], [0.8, 0.8]]
widths = [0.2, 0.2, 0.2]
weights = [1.0, 0.5, 1.0]
multihills = Multihills(centers, widths, weights)

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
        n_iteration = 40,
        n_initial_points = 5,
        bounds_design = multihills_bounds,
        bounds_action = multihills_bounds,
        n_dim_design = 2,
        n_dim_action = 2,
        n_actions = 2,
        n_ys = args.n_ys,
        n_restarts = args.n_restarts,
        acq_opt_iter = args.acq_opt_iter,
        acq_opt_lr = args.acq_opt_lr,
        optimizer = args.optimizer,
        policy_transfer = args.policy_transfer,
        n_resampling = args.n_resampling,
        n_candidates = 1,
        func_is_noisy = False,
        func_noise = 0.1,
    )

    # --- nn-specific below
    config.n_layers = args.n_layers
    config.activation = args.activation
    config.hidden_coeff = args.hidden_coeff
    
    # -- non-myopic-specific below
    config.amortized_vi = args.amortized_vi
    config.lookahead_steps = args.lookahead_steps

    # --- app-specific below
    config.app = 'topk'
    config.algo = 'hes'
    config.fname = 'multihills'
    config.dist_weight = 20.0
    config.dist_threshold = 0.5
    config.save_dir = f'experiments/opt_{config.app}_{config.fname}_{config.algo}_seed{config.seed}_n_ys{config.n_ys}_n_restarts{config.n_restarts}_amortized_vi{config.amortized_vi}_lookahead_steps{config.lookahead_steps}_n_layers{config.n_layers}_activation{config.activation}_hidden_coeff{config.hidden_coeff}_acq_opt_lr{config.acq_opt_lr}_acq_opt_iter{config.acq_opt_iter}_n_resampling{config.n_resampling}'

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

    # --- Set plot settings
    ax.set(
        xlabel='$x_1$', ylabel='$x_2$', xlim=multihills_bounds, ylim=multihills_bounds
    )

    # --- save plot and close
    neatplot.save_figure(f'topk_{iteration}', 'png', config.save_dir)
    plt.close()


if __name__ == "__main__":
    run_script()
