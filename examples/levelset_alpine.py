import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import torch
from botorch.acquisition import PosteriorMean
from src.run import run_hes_trial, run_gpclassifier_trial, uniform_random_sample_domain
from alpine import AlpineN1, plot_alpine_2d
import neatplot
from argparse import Namespace, ArgumentParser
import pickle
import copy
from src.hentropy import qHEntropySearch
from botorch.models import SingleTaskGP

neatplot.set_style()
neatplot.update_rc('axes.grid', False)
neatplot.update_rc('text.usetex', False)

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--num_dim", type=int, default=2)
parser.add_argument("--algo", type=str, default='hes')
args = parser.parse_args()

# Initialize Alpine function
alpine = AlpineN1(d=args.num_dim, x_scale=0.5, y_scale=0.05)
alpine_bounds = alpine.get_bounds()
print("Bounds", alpine_bounds)


def run_script():
    # Configure hes trial
    config = Namespace(
        seed=args.seed,
        num_iteration=100,
        num_initial_points=5,
        bounds_design=alpine_bounds,
        bounds_action=alpine_bounds,
        num_dim_design=args.num_dim,
        num_dim_action=1,
        num_action=None,
        num_outer_mc=32,
        num_restarts=16,
        acq_opt_iter=100,
        acq_opt_lr=0.1,
        num_candidates=1,
        func_is_noisy=False,
        func_noise=0.1,
    )
    # --- app-specific below
    config.app = 'levelset'
    config.algo = args.algo
    config.fname = 'alpine'
    config.support_num_per_dim = 15
    config.levelset_threshold = 0.45
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
            plot_levelset_alpine,
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
    """Alpine function with torch tensor input/output."""
    y = alpine(x, tensor=True)
    return y


@np.vectorize
def func_vec(x, y):
    """Vectorized alpine for contour plot. Return f on input = (x, y)."""
    inp = np.array([x, y]).reshape(-1)
    return alpine(inp)


def plot_function_contour(ax, threshold=None):
    gridwidth = 0.1
    plot_alpine_2d(ax=ax, threshold=threshold)


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
    # ax.set(title=title)
    ax.set_title(label=title, fontdict={'fontsize': 25})


def plot_levelset_alpine(config, next_x, data, action_samples, iteration):
    if iteration % 5 != 0 and iteration != 1:
        return
    fig, ax = plt.subplots(figsize=(6, 6))

    plot_function_contour(ax, threshold=config.levelset_threshold / alpine.y_scale)
    plot_data(ax, data)
    plot_next_query(ax, next_x)
    plot_settings(ax, config)

    # Save plot and close
    neatplot.save_figure(f'topk_{iteration}', 'pdf', config.save_dir)
    plt.close()


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
