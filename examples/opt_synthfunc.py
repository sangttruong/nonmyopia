#!/usr/bin/env python
"""Non-myopic H-entropy search for synthetic function"""

from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch import fit_gpytorch_model
from botorch.sampling.samplers import SobolQMCNormalSampler
import neatplot
from synthfunc import SynthFunc
from src.utilities import get_init_data
from src.run import *
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy
from argparse import Namespace, ArgumentParser
__author__ = ""
__copyright__ = "Copyright 2022, Stanford University"


import torch
import numpy as np


neatplot.set_style()
neatplot.update_rc("axes.grid", False)
neatplot.update_rc("font.size", 16)
neatplot.update_rc("text.usetex", False)

parser = ArgumentParser()

# general arguments
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--seed_synthfunc", type=int, default=1)
parser.add_argument("--dataset", type=str, default="synthetic")
parser.add_argument("--n_dim", type=int, default=2)
parser.add_argument("--n_actions", type=int, default=1)
parser.add_argument("--algo", type=str, default="hes_mc")
parser.add_argument("--gpuid", type=int, default=0)
parser.add_argument("--lookahead_steps", type=int, default=1)
parser.add_argument("--n_initial_points", type=int, default=10)
parser.add_argument("--r", type=float)
parser.add_argument("--local_init", choices=("True", "False"), default="True")
parser.add_argument("--n_iterations", type=int, default=100)

# MC approximation
parser.add_argument("--n_samples", type=int, default=16)
parser.add_argument("--decay_factor", type=int, default=1)

# optimizer
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--acq_opt_iter", type=int, default=1000)
parser.add_argument("--n_restarts", type=int, default=128)

# amortization
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--activation", type=str, default="elu")
parser.add_argument("--hidden_coeff", type=float, default=4)

# baseline
parser.add_argument("--baseline", choices=("True", "False"), default="False")
parser.add_argument("--baseline_n_layers", type=int, default=2)
parser.add_argument("--baseline_hidden_coeff", type=int, default=1)
parser.add_argument("--baseline_activation", type=str, default="relu")
parser.add_argument("--baseline_lr", type=float, default=0.1)

parser.add_argument("--init_noise_thredhold", type=float, default=0.01)

# resampling
parser.add_argument("--n_resampling_max", type=int, default=1)
parser.add_argument("--n_resampling_improvement_threadhold",
                    type=float, default=0.01)
"""When n_resampling_max == 1 and n_resampling_improvement_threadhold is small, we have 
the orange curve. n_resampling_max is large and n_resampling_improvement_threadhold is
large, we have the pink curve (closer to stochastic gradient descent). We can interpolate
between these 2 options by setting both hyperparameters to some moderate value. """

# gp hyperparameters
parser.add_argument(
    "--learn_hypers", choices=("True", "False"), default="False")

# patients
parser.add_argument("--max_patient", type=int, default=5000)
parser.add_argument("--max_patient_resampling", type=int, default=5)

# annealing for hes optimizer
parser.add_argument("--eta_min", type=float, default=0.0001)
"""When eta_min = acq_opt_lr, the learning rate is constant at acq_opt_lr"""
parser.add_argument("--T_max", type=int, default=100)
"""large T_max corresponds to slow annealing"""

# get init data
parser.add_argument("--path_str", type=str)
parser.add_argument("--start_iter", type=int)

# Check parameter
parser.add_argument("--mode", choices=("train", "check"), default="train")
parser.add_argument("--check_dir", type=str, default="experiments")

args = parser.parse_args()

# Fix arg types, set defaults, perform checks
args.local_init = True if args.local_init == "True" else False
args.baseline = True if args.baseline == "True" else False
args.learn_hypers = True if args.learn_hypers == "True" else False
assert args.lookahead_steps > 0

# Initialize synthetic function
hypers = {"ls": 0.1, "alpha": 2.0, "sigma": 1e-2, "n_dimx": args.n_dim}
args.hypers = hypers if not args.learn_hypers else None

# TODO: need to figure out the bound for input in chemical dataset
sf_bounds = [-1, 1]
sf_domain = [sf_bounds] * args.n_dim

if args.dataset == "synthetic":
    sf = SynthFunc(seed=args.seed_synthfunc, hypers=hypers, n_obs=50, domain=sf_domain)

    @np.vectorize
    def sf_vec(x, y):
        """Return f on input = (x, y)."""
        return sf((x, y))


    def func(x_list):
        """Synthetic function with torch tensor input/output."""
        x_list = [xi.cpu().detach().numpy().tolist() for xi in x_list]
        x_list = np.array(x_list).reshape(-1, args.n_dim)
        y_list = [sf(x) for x in x_list]
        y_list = y_list[0] if len(y_list) == 1 else y_list
        y_tensor = torch.tensor(np.array(y_list).reshape(-1, 1))
        return y_tensor

elif args.dataset == "chemical":
    with open("examples/semisynthetic.pt", 'rb') as file_handle:
        sf = pickle.load(file_handle)

    @np.vectorize
    def sf_vec(x, y):
        """Return f on input = (x, y)."""
        return sf.predict([[x, y]])


    def func(x_list):
        """Synthetic function with torch tensor input/output."""
        x_list = [xi.cpu().detach().numpy().tolist() for xi in x_list]
        x_list = np.array(x_list).reshape(-1, args.n_dim)
        y_list = sf.predict(x_list)
        y_list = y_list[0] if len(y_list) == 1 else y_list
        y_tensor = torch.tensor(np.array(y_list).reshape(-1, 1))
        return y_tensor

else: raise NotImplemented

def plot_synthfunc_2d(ax):
    """Plot synthetic function in 2d."""
    domain_plot = sf_domain
    grid = 0.01
    xpts = np.arange(domain_plot[0][0], domain_plot[0][1], grid)
    ypts = np.arange(domain_plot[1][0], domain_plot[1][1], grid)
    X, Y = np.meshgrid(xpts, ypts)
    Z = sf_vec(X, Y)
    cf = ax.contourf(X, Y, Z, 20, cmap=cm.GnBu, zorder=0)
    plot = ax.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
    cbar = plt.colorbar(cf, fraction=0.046, pad=0.04)
    # add colorbar here


def plot_function_contour(ax):
    plot_synthfunc_2d(ax)


def plot_data(ax, data):
    data_x = copy.deepcopy(data.x.cpu().detach()).numpy()
    for xi in data_x:
        ax.plot(xi[0], xi[1], "o", color="black", markersize=2)


def plot_next_query(ax, next_x):
    next_x = copy.deepcopy(next_x.cpu().detach()).numpy().reshape(-1)
    ax.plot(next_x[0], next_x[1], "o", color="deeppink", markersize=2)


def plot_settings(ax, config):
    bounds_plot = sf_bounds
    bounds_plot_ext = [bounds_plot[0] - 0.05, bounds_plot[1] + 0.05]
    ax.set(xlabel="$x_1$", ylabel="$x_2$",
           xlim=bounds_plot_ext, ylim=bounds_plot_ext)

    # Set title
    if config.algo == "hes_vi":
        title = "HES VI"
    if config.algo == "hes_mc":
        title = "HES MC"
    if config.algo == "random":
        title = "Random"
    if config.algo == "qEI":
        title = "Expected Improvement"
    if config.algo == "qPI":
        title = "Probability of Improvement"
    if config.algo == "qSR":
        title = "Simple Regret"
    if config.algo == "qUCB":
        title = "Upper Confident Bound"
    if config.algo == "kg":
        title = "Knowledge Gradient"
    if config.algo == "rs":
        title = "Random Search"
    if config.algo == "us":
        title = "Uncertainty Sampling"
    # ax.set(title=title)
    ax.set_title(label=title, fontdict={"fontsize": 25})


def plot_action_samples(ax, action_samples, config):
    action_samples = copy.deepcopy(action_samples.cpu().detach()).numpy()
    action_samples = action_samples.reshape(-1,
                                            config.n_actions, config.n_dim_action)
    for x_actions in action_samples:
        lines2d = ax.plot(x_actions[0][0], x_actions[0]
                          [1], "v", color="b", alpha=0.5, markersize=4)
        if config.n_actions >= 2:
            color = lines2d[0].get_color()
            ax.plot(x_actions[1][0], x_actions[1][1], "v",
                    color="b", alpha=0.5, markersize=4)
            line_1_x = [x_actions[0][0], x_actions[1][0]]
            line_1_y = [x_actions[0][1], x_actions[1][1]]
            ax.plot(line_1_x, line_1_y, "--", color=color)
        if config.n_actions >= 3:
            ax.plot(x_actions[2][0], x_actions[2][1], "v",
                    color="b", alpha=0.5, markersize=4)
            line_2_x = [x_actions[1][0], x_actions[2][0]]
            line_2_y = [x_actions[1][1], x_actions[2][1]]
            ax.plot(line_2_x, line_2_y, "--", color=color)


def plot_optimal_action(ax, optimal_action, config):
    optimal_action = optimal_action.reshape(-1, 2)
    for x_action in optimal_action:
        x_action = x_action.squeeze()
        ax.plot(x_action[0], x_action[1], "*", mfc="gold",
                mec="darkgoldenrod", markersize=4)


def plot_groundtruth_optimal_action(ax, config):

    if config.n_actions <= 3:
        centers = [[4.0, 4.0], [2.45, 4.0], [4.0, 2.45]]
    elif config.n_actions == 5:
        centers = [[4.0, 4.0], [2.45, 4.0], [
            4.0, 2.45], [1.0, 4.0], [4.0, 1.0]]
    gt_optimal_action = np.array(centers)

    for x_action in gt_optimal_action:
        ax.plot(x_action[0], x_action[1], "s", color="blue", markersize=4)


def plot_spotlight(ax, config, previous_x):
    previous_x = previous_x.squeeze()
    previous_x = previous_x.cpu().detach().numpy()
    splotlight = plt.Rectangle(
        (previous_x[0]-config.r, previous_x[1]-config.r),
        2*config.r, 2*config.r,
        color="b", fill=False
    )
    ax.add_patch(splotlight)


def plot_topk(config, data, iteration, next_x, previous_x, actions):
    """Plotting for topk."""
    if iteration in config.plot_iters:
        fig, ax = plt.subplots(figsize=(6, 6))

        plot_function_contour(ax)
        plot_data(ax, data)
        plot_action_samples(ax, actions, config)
        plot_next_query(ax, next_x)
        if config.r:
            plot_spotlight(ax, config, previous_x)
        plot_settings(ax, config)

        # Save plot and close
        neatplot.save_figure(f"topk_{iteration}", "png", config.save_dir)
        plt.close()


def eval_topk(config, data, iteration, next_x, previous_x):
    """Return evaluation metric."""

    c = config
    noise = torch.rand(
        [c.n_restarts, c.n_actions, c.n_dim_action], device=c.device)
    bayes_actions = c.bounds_action[0] + \
        (c.bounds_action[1]-c.bounds_action[0])*noise
    bayes_actions.requires_grad_(True)
    sampler = SobolQMCNormalSampler(
        num_samples=c.n_samples, resample=False, collapse_batch_dims=True)
    mll_hes, model_hes = initialize_model(
        data, covar_module=ScaleKernel(base_kernel=RBFKernel()))

    if not config.learn_hypers:
        print(
            f"config.learn_hypers={config.learn_hypers}, using hypers from config.hypers")
        model_hes.covar_module.base_kernel.lengthscale = [
            [config.hypers["ls"]]]
        # NOTE: GPyTorch outputscale should be set to the SynthFunc alpha squared
        model_hes.covar_module.outputscale = config.hypers["alpha"]**2
        model_hes.likelihood.noise_covar.noise = [config.hypers["sigma"]]

        model_hes.covar_module.base_kernel.raw_lengthscale.requires_grad_(
            False)
        model_hes.covar_module.raw_outputscale.requires_grad_(False)
        model_hes.likelihood.noise_covar.raw_noise.requires_grad_(False)
    fit_gpytorch_model(mll_hes)

    optim = torch.optim.Adam([bayes_actions], lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=c.T_max, eta_min=c.eta_min)

    patient = c.max_patient
    min_loss = float("inf")
    losses = []
    lrs = []
    for _ in tqdm(range(c.acq_opt_iter)):
        p_yi_xiDi = model_hes.posterior(torch.tanh(bayes_actions))
        batch_yis = sampler(p_yi_xiDi)
        batch_yis = batch_yis.mean(dim=0)
        result = batch_yis.squeeze()
        loss = -result.sum()

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        lrs.append(scheduler.get_last_lr())
        losses.append(loss.cpu().detach().numpy())

        if loss < min_loss:
            min_loss = loss
            patient = c.max_patient
            best_restart = torch.argmax(batch_yis)
            optimal_action = (torch.tanh(bayes_actions))[
                best_restart, :, :].cpu().detach().numpy()
            eval_metric = result[best_restart].cpu().detach().numpy()
        else:
            patient -= 1

        if patient < 0:
            break

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(np.array(losses), "b-", linewidth=1)
    ax2.plot(np.array(lrs), "r-", linewidth=1)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="b")
    ax2.set_ylabel("Learning rate", color="r")
    if not os.path.exists(f"{c.save_dir}/{c.algo}"):
        os.makedirs(f"{c.save_dir}/{c.algo}")
    plt.savefig(
        f"{c.save_dir}/{c.algo}/acq_opt_eval_{iteration}.png", bbox_inches="tight")

    # Plot optimal_action in special eval plot here
    if iteration in config.plot_iters:
        _, ax = plt.subplots(figsize=(6, 6))

        plot_function_contour(ax)
        plot_data(ax, data)
        #plot_groundtruth_optimal_action(ax, config)
        plot_next_query(ax, next_x)
        if config.r:
            plot_spotlight(ax, config, previous_x)
        plot_optimal_action(ax, optimal_action, config)
        plot_settings(ax, config)

        if not os.path.exists(f"{config.save_dir}/{c.algo}/"):
            os.makedirs(f"{config.save_dir}/{c.algo}/")

        # Save plot and close
        neatplot.save_figure(
            f"topk_eval_{iteration}", "png", f"{config.save_dir}/{c.algo}/")
        plt.close()

    # Return eval_metric and optimal_action (or None)
    return eval_metric, optimal_action


if __name__ == "__main__":

    # Configure hes trial
    device = f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    config = Namespace(
        device=device,
        torch_dtype=torch.double,
        seed=args.seed,
        seed_synthfunc=args.seed_synthfunc,
        n_iterations=args.n_iterations,
        n_initial_points=args.n_initial_points,
        bounds_design=sf_bounds,
        bounds_action=sf_bounds,
        n_dim_design=args.n_dim,
        n_dim_action=args.n_dim,
        n_actions=args.n_actions,
        n_samples=args.n_samples,
        n_restarts=args.n_restarts,
        acq_opt_iter=args.acq_opt_iter,
        optimizer=args.optimizer,
        n_resampling_max=args.n_resampling_max,
        n_resampling_improvement_threadhold=args.n_resampling_improvement_threadhold,
        baseline=args.baseline,
        n_candidates=1,
        func=sf,
        func_is_noisy=False,
        func_noise=0.1,
        plot_iters=list(range(0, 101, 1)),
        learn_hypers=args.learn_hypers,
        hypers=args.hypers,
        max_patient=args.max_patient,
        max_patient_resampling=args.max_patient_resampling,
        eta_min=args.eta_min,
        T_max=args.T_max,
        start_iter=1,
        r=args.r,
        local_init=args.local_init,
        decay_factor=args.decay_factor,
        init_noise_thredhold=args.init_noise_thredhold,
        dataset=args.dataset,
    )

    # --- nn-specific below
    config.n_layers = args.n_layers
    config.activation = args.activation
    config.hidden_coeff = args.hidden_coeff

    # -- non-myopic-specific below
    config.lookahead_steps = args.lookahead_steps

    # --- app-specific below
    config.app = "topk"
    config.fname = "synthfunc"
    config.algo = args.algo
    config.dist_weight = 20.0
    config.dist_threshold = 2.5
    config.save_dir = f"experiments/"\
    f"opt{config.app}_"\
    f"fname{config.fname}_"\
    f"algo{config.algo}_"\
    f"seed{config.seed}_"\
    f"n_samples{config.n_samples}_"\
    f"n_restarts{config.n_restarts}_"\
    f"lookahead_steps{config.lookahead_steps}_"\
    f"n_layers{config.n_layers}_"\
    f"activation{config.activation}_"\
    f"hidden_coeff{config.hidden_coeff}_"\
    f"acq_opt_iter{config.acq_opt_iter}_"\
    f"n_resampling_max{config.n_resampling_max}_"\
    f"baseline{config.baseline}_"\
    f"n_initial_points{config.n_initial_points}_"\
    f"r{args.r}"
    config.check_dir = args.check_dir

    # Set any initial data
    if args.start_iter is not None:
        config.init_data = get_init_data(
            path_str=config.save_dir, start_iter=args.start_iter, n_init_data=0)
    else:
        config.init_data = None

    # Run hes trial
    make_save_dir(config)

    init_data = copy.deepcopy(config.init_data)
    config.init_data = init_data

    if config.algo == "hes_vi":
        config.vi = True
        config.acq_opt_lr = 0.001
    elif config.algo == "hes_mc":
        config.vi = False
        config.acq_opt_lr = 0.1

    run_hes_trial(
        func=func,
        config=config,
        plot_function=None,
        eval_function=eval_topk,
        final_eval_function=None,
    )
