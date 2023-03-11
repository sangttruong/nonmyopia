from argparse import Namespace, ArgumentParser
import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import torch
from botorch.acquisition import PosteriorMean
from src.run import run_hes_trial, run_gpclassifier_trial, uniform_random_sample_domain
from multihills import Multihills, multihills_bounds
import neatplot
import pickle
from src.hentropy import qHEntropySearch
from botorch.models import SingleTaskGP

neatplot.set_style()
neatplot.update_rc('axes.grid', False)
neatplot.update_rc('text.usetex', False)
neatplot.update_rc('font.size', 16)

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--algo", type=str, default='hes')
args = parser.parse_args()

# Initialize Multihills
centers = [[0.2, 0.15], [0.2, 0.75], [0.75, 0.5]]
widths = [0.2, 0.2, 0.2]
weights = [0.4, 0.8, 0.9]
multihills = Multihills(centers, widths, weights)


def run_script():
    # Configure hes trial
    config = Namespace(
        seed=args.seed,
        num_iteration=40,
        num_initial_points=5,
        bounds_design=multihills_bounds,
        bounds_action=multihills_bounds,
        num_dim_design=2,
        num_dim_action=1,
        num_action=None,
        num_outer_mc=32,
        num_restarts=16,
        acq_opt_iter=200,
        acq_opt_lr=0.1,
        num_candidates=1,
        func_is_noisy=False,
        func_noise=0.1,
    )
    # --- app-specific below
    config.app = 'levelset'
    config.algo = args.algo
    config.fname = 'multihills'
    config.support_num_per_dim = 10
    config.levelset_threshold = 2.1
    config.num_action = config.support_num_per_dim ** 2
    config.batch_size = config.num_restarts
    config.support_points = get_support_points(config)
    config.save_dir = f'experiments/{config.app}_{config.fname}_{config.algo}_seed{config.seed}'

    # Run hes trial
    if config.algo == 'gpclassifier':
        run_gpclassifier_trial(func,
                               config,
                               None,
                               eval_levelset,
                               final_eval_levelset)
    else:
        run_hes_trial(
            func,
            config,
            None,
            eval_levelset,
            final_eval_levelset,
        )


def get_support_points(config):
    """Return support points based on config."""
    assert config.num_dim_design == 2
    domain = [config.bounds_design] * config.num_dim_design

    xpts = np.linspace(domain[0][0] + 0.05, domain[0][1] - 0.05, config.support_num_per_dim)
    ypts = np.linspace(domain[1][0] + 0.05, domain[1][1] - 0.05, config.support_num_per_dim)

    support_points = [[x, y] for x in xpts for y in ypts]
    support_points = torch.tensor(support_points)

    return support_points


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


def plot_function_contour(ax, threshold=None):
    gridwidth = 0.01
    xpts = np.arange(multihills_bounds[0], multihills_bounds[1], gridwidth)
    ypts = np.arange(multihills_bounds[0], multihills_bounds[1], gridwidth)
    X, Y = np.meshgrid(xpts, ypts)
    Z = func_vec(X, Y)
    ax.contour(X, Y, Z, 50, cmap=cm.Greens_r, zorder=0)
    print(Z.max(), Z.min(), Z.mean())
    if threshold:
        ax.contour(X, Y, Z, levels=[threshold], colors='red', linewidths=3, linestyles='dashed')


def eval_levelset(qhes, config, data, next_x, iteration, random=False):
    """
    Return evaluation metric. If random=True, use random uniform test; otherwise use grid test.
    """
    with torch.no_grad():
        if type(qhes) is qHEntropySearch:
            postmean = PosteriorMean(qhes.model)
        else:
            assert type(qhes) is SingleTaskGP
            postmean = PosteriorMean(qhes)  # the first argument is directly a GP model

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
    run_script()

    # Pick a level-set threshold
    # fig, ax = plt.subplots(figsize=(6, 6))
    # plot_function_contour(ax, 2.1)
    # plt.show()
