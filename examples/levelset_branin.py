from argparse import Namespace

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import torch
from botorch.acquisition import PosteriorMean

from src.run import run_hes_trial, uniform_random_sample_domain

from branin import (
    branin,
    branin_normalized,
    transform_to_branin_domain,
)
from branin import domain as branin_domain
import neatplot
import os
import argparse
import pickle

neatplot.set_style()
neatplot.update_rc('axes.grid', False)
neatplot.update_rc('text.usetex', False)


def run_script(args):
    # Configure hes trial
    config = Namespace(
        seed=args.seed,
        num_iteration=30,
        num_initial_points=5,
        bounds_design=[0.0, 1.0],
        bounds_action=[0.0, 1.0],
        num_dim_design=2,
        num_dim_action=1,
        num_action=None,
        num_outer_mc=16,
        num_restarts=64,
        acq_opt_iter=100,
        acq_opt_lr=0.1,
        num_candidates=1,
        func_is_noisy=False,
        func_noise=0.1,
    )
    # --- app-specific below
    config.app = args.app
    config.algo = args.algo
    config.fname = 'branin'
    config.support_num_per_dim = 10
    config.levelset_threshold = -1
    config.num_action = config.support_num_per_dim ** 2
    config.batch_size = config.num_restarts
    config.support_points = get_support_points(config)
    config.save_dir = f'experiments/{config.app}_{config.fname}_{config.algo}_seed{config.seed}'

    # Run hes trial
    run_hes_trial(
        func,
        config,
        plot_levelset_branin,
        eval_levelset,
        final_eval_levelset,
    )


def get_support_points(config):
    """Return support points based on config."""
    assert config.num_dim_design == 2
    domain = [config.bounds_design] * config.num_dim_design

    diffs = (domain[0][1] - domain[0][0], domain[1][1] - domain[1][0])
    starts = (domain[0][0] + 0.05, domain[1][0] + 0.05)
    ends = (domain[0][1], domain[1][1])

    xpts = np.arange(starts[0], ends[0], diffs[0] / config.support_num_per_dim)
    ypts = np.arange(starts[1], ends[1], diffs[1] / config.support_num_per_dim)

    support_points = [[x, y] for x in xpts for y in ypts]
    support_points = torch.tensor(support_points)

    return support_points


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


def plot_levelset_branin(config, next_x, data, action_samples, iteration):
    """Plotting for topk."""
    if iteration % 5 != 0 and iteration != 1:
        return
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
    support_points = transform_to_branin_domain(config.support_points)
    x_actions = action_samples[0]  # only plot the result from the first fantasy model
    for i in range(config.num_action):
        color = 'red' if x_actions[i][0] > 0.5 else 'blue'
        ax.plot(support_points[i][0], support_points[i][1], 'o', color=color, markersize=2)

    # --- Set plot settings
    ax.set(xlabel='$x_1$', ylabel='$x_2$', xlim=[-5.3, 10.3], ylim=[-0.3, 15.3])

    # --- save plot and close
    neatplot.save_figure(f'levelset_{iteration}', 'png', config.save_dir)
    plt.close()

    # Save data for further plotting
    if iteration == config.num_iteration:
        save_data = [(X, Y, Z), data_x, support_points, x_actions]
        pickle.dump(save_data, open(f'{config.save_dir}/plot_data.pkl', 'wb'))


def eval_levelset(qhes, config, data, next_x, iteration, random=False):
    """
    Return evaluation metric. If random=True, use random uniform test; otherwise use grid test.
    """
    with torch.no_grad():
        postmean = PosteriorMean(qhes.model)
        test_samples, test_targets = get_test_samples_targets(func, config, random)
        test_pm = postmean(test_samples.unsqueeze(1))  # batch_size x 1 x data_dim
        test_pm = test_pm.detach().numpy()

        test_preds = []
        for val in test_pm:
            test_preds.append(1.0 if val > config.levelset_threshold else 0.0)

        test_preds = np.array(test_preds)
        acc = 1 - np.abs(test_targets - test_preds).mean()
        acc = acc.tolist()

        # Return eval_metric and eval_data (or None)
        return acc, None


def get_test_samples_targets(func, config, random=False):
    """
    Return test set samples and targets for LSE evaluation.
    If random=True, use random uniform test; otherwise use grid test.
    """
    domain = [config.bounds_design] * config.num_dim_design
    if random:
        test_samples = uniform_random_sample_domain(domain, 1000)
    else:
        test_samples = config.support_points
    test_samples_values = func(test_samples).squeeze().detach().numpy()
    test_targets = []
    for val in test_samples_values:
        test_targets.append(1.0 if val > config.levelset_threshold else 0.0)
    test_targets = np.array(test_targets)
    return test_samples, test_targets


def final_eval_levelset(eval_list, config):
    """Final evaluation after BO."""
    print(eval_list)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(eval_list)
    ax.set(xlabel='Iteration', ylabel='Accuracy')
    neatplot.save_figure('levelset_final', 'png', config.save_dir)
    plt.close()
    pickle.dump(eval_list, open(f'{config.save_dir}/eval_list.pkl', 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seed", default=10)
    parser.add_argument('--app', type=str, help='levelset, topk', default='levelset')
    parser.add_argument('--algo', type=str, help='hes', default='hes')
    args = parser.parse_args()
    run_script(args)
