#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Evaluate and plot."""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from argparse import ArgumentParser

import gpytorch
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from draw_metrics import get_env_info
from gpytorch.mlls import ExactMarginalLogLikelihood
from main import Parameters
from tueplots import bundles
from utils import eval_func, make_env, set_seed, str2bool

plt.rcParams.update(bundles.neurips2024())

parser = ArgumentParser()
parser.add_argument("--setting", type=str, default="default")
parser.add_argument("--n_initial_points", type=int, default=-1)
parser.add_argument("--n_restarts", type=int, default=64)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--kernel", type=str, default="RBF")
parser.add_argument("--result_dir", type=str, default="./results")
args = parser.parse_args()

if args.setting == "default":
    algos_name = [
        "SR",
        "EI",
        "PI",
        "UCB",
        "KG",
        "MSL",
        "Ours",
    ]

    algos = [
        "qSR",
        "qEI",
        "qPI",
        "qUCB",
        "qKG",
        "HES-TS-20",
        "HES-TS-AM-20",
    ]

    seeds = {
        "qSR": 2,
        "qEI": 3,
        "qPI": 5,
        "qUCB": 2,
        "qKG": 3,
        "HES-TS-20": 2,
        "HES-TS-AM-20": 5,
    }

    env_names = [
        "Ackley",
        # "Ackley4D",
        "Alpine",
        # "Beale",
        # "Branin",
        # "Cosine8",
        # "EggHolder",
        # "Griewank",
        # "Hartmann",
        # "HolderTable",
        # "Levy",
        # "Powell",
        # "SixHumpCamel",
        # "StyblinskiTang",
        "SynGP",
    ]

    env_noises = [
        0.0,
        0.01,
        0.05,
    ]

    env_discretizeds = [
        False,
    ]

    cost_functions = [
        # "euclidean",
        # "manhattan",
        "r-spotlight",
        # "non-markovian"
    ]

elif args.setting == "lookahead":
    from ablation_configs.lookahead import algos_name, algos, seeds, env_names, env_noises, env_discretizeds, cost_functions
elif args.setting == "restart":
    from ablation_configs.restart import algos_name, algos, seeds, env_names, env_noises, env_discretizeds, cost_functions
elif args.setting == "network":
    from ablation_configs.network import algos_name, algos, seeds, env_names, env_noises, env_discretizeds, cost_functions
elif args.setting == "kernel":
    from ablation_configs.kernel import algos_name, algos, seeds, env_names, env_noises, env_discretizeds, cost_functions
elif args.setting == "ucb":
    from ablation_configs.ucb import algos_name, algos, seeds, env_names, env_noises, env_discretizeds, cost_functions
elif args.setting == "init_samples_ackley":
    from ablation_configs.init_samples_ackley import algos_name, algos, seeds, env_names, env_noises, env_discretizeds, cost_functions
elif args.setting == "init_samples_alpine":
    from ablation_configs.init_samples_alpine import algos_name, algos, seeds, env_names, env_noises, env_discretizeds, cost_functions
elif args.setting == "init_samples_syngp":
    from ablation_configs.init_samples_syngp import algos_name, algos, seeds, env_names, env_noises, env_discretizeds, cost_functions
elif args.setting == "init_samples_nightlight":
    from ablation_configs.init_samples_nightlight import algos_name, algos, seeds, env_names, env_noises, env_discretizeds, cost_functions
elif args.setting == "nightlight":
    from ablation_configs.nightlight import algos_name, algos, seeds, env_names, env_noises, env_discretizeds, cost_functions

cost_function_names = {
    "euclidean": "Euclidean cost",
    "manhattan": "Manhattan cost",
    "r-spotlight": "$r$-spotlight cost",
    "non-markovian": "Non-Markovian cost",
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_results(seed, en, cost_fn, args):
    outputs = {}

    for env_name in env_names:
        for algo in algos:
            seed = seeds[algo]
            if env_name == "Ackley" and args.setting == "ucb" and algo == "qUCB-50":
                seed = 72
            x_dim, bounds, radius, n_initial_points, algo_n_iterations = get_env_info(
                env_name, device
            )
            
            if args.n_initial_points > -1:
                difference = args.n_initial_points - n_initial_points
                n_initial_points = args.n_initial_points
                algo_n_iterations += difference

            print(f"Reading {env_name} - {args.kernel} - {algo} - {n_initial_points} - {args.hidden_dim} - {args.n_restarts}")
            base_path = (
                f"{args.result_dir}/{env_name}_{en}{'_' + args.kernel if args.kernel != 'RBF' else ''}/"
                f"{algo}_{cost_fn}_seed{seed}_init{n_initial_points}_hidden{args.hidden_dim}_rs{args.n_restarts}"
            )
            
            surr_file = f"{base_path}/surr_model_{algo_n_iterations-1}.pt"
            buffer_file = (
                f"{base_path}/buffer.pt"
            )

            buffer = torch.load(buffer_file, map_location=device).to(
                dtype=torch.float64
            )
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=gpytorch.priors.NormalPrior(0, 1e-2)
            )
            if args.kernel == "RBF":
                kernel = None
            elif args.kernel.startswith("Matern"):
                nu = float(args.kernel.split("-")[-1])
                kernel = gpytorch.kernels.MaternKernel(nu=nu)
            elif args.kernel == "Linear":
                kernel = gpytorch.kernels.LinearKernel()
                
            surr_model = SingleTaskGP(
                buffer["x"][: algo_n_iterations + 1],
                buffer["y"][: algo_n_iterations + 1],
                likelihood=likelihood,
                covar_module=kernel,
            ).to(device, dtype=torch.float64)
            
            mll = ExactMarginalLogLikelihood(surr_model.likelihood, surr_model)
            fit_gpytorch_model(mll)

            outputs[(env_name, algo)] = [buffer, surr_model]

    return outputs


class PseudoArgs:
    def __init__(self, args):
        self.seed = 2
        self.task = "topk"
        self.env_name = "SynGP"
        self.env_noise = 0.01
        self.env_discretized = False
        self.algo = "HES-TS-AM-20"
        self.cost_fn = "r-spotlight"
        self.plot = False
        self.gpu_id = 0
        self.cont = False
        self.setting = args.setting
        self.result_dir = args.result_dir
        self.n_initial_points = args.n_initial_points
        self.n_restarts = args.n_restarts
        self.hidden_dim = args.hidden_dim
        self.kernel = args.kernel


def draw_result(results, seed, en, cost_fn, args):
    n_space = 100
    fig, axs = plt.subplots(
        len(env_names), len(algos), figsize=(4 * len(algos), 4 * len(env_names))
    )
    cmap = cm.summer  # Choose a colormap

    for env_name in env_names:
        # Row: env
        # Col: algo
        x_dim, bounds, radius, n_initial_points, algo_n_iterations = get_env_info(
            env_name, device
        )
        if args.n_initial_points > -1:
            difference = args.n_initial_points - n_initial_points
            n_initial_points = args.n_initial_points
            algo_n_iterations += difference
                
        env = make_env(
            name=env_name,
            x_dim=x_dim,
            bounds=bounds,
            noise_std=en,
        )
        env = env.to(
            dtype=torch.float64,
            device=device,
        )

        total_steps = algo_n_iterations - n_initial_points
        norm = mcolors.Normalize(vmin=0, vmax=total_steps)
        colors = [cmap(1 - norm(lhi)) for lhi in range(total_steps+1)]

        for algo in algos:
            seed = seeds[algo]

            buffer, surr_model = results[(env_name, algo)]
            col_idx = algos.index(algo)
            row_idx = env_names.index(env_name)
            if len(env_names) == 1:
                if len(algos) == 1:
                    ax = axs
                else:
                    ax = axs[col_idx]
            elif len(algos) == 1:
                ax = axs[row_idx]
            else:
                ax = axs[row_idx][col_idx]

            # Parse args
            args = PseudoArgs(args)
            args.seed = int(seed)
            args.env_name = env_name
            args.env_noise = en
            args.algo = algo
            args.seed = cost_fn

            set_seed(seed)
            parms = Parameters(args)
            real_loss, bayes_action = eval_func(
                env, surr_model, parms, buffer, algo_n_iterations - 1, None
            )

            data_x = buffer["x"].cpu().detach()

            # Plotting ###############################################################
            bounds_plot_x, bounds_plot_y = np.array([0, 1]), np.array([0, 1])
            ax.set(xlim=bounds_plot_x, ylim=bounds_plot_y)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(axis="both", labelsize=16)

            # Plot function in 2D ####################################################
            X_domain, Y_domain = bounds_plot_x, bounds_plot_y
            
            if env_name != "NightLight":
                X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
                X, Y = np.meshgrid(X, Y)
                XY = torch.tensor(np.array([X, Y]))  # >> 2 x 100 x 100
                Z = env(XY.to(parms.device, parms.torch_dtype).reshape(2, -1).T).reshape(
                    X.shape
                )
                Z = Z.cpu().detach()
                ax.contourf(X, Y, Z, levels=30, cmap="bwr", alpha=0.7)
            elif env_name == "NightLight":
                ax.imshow(
                    env.env.f.cpu().detach().numpy(),
                    cmap="gray",
                    extent=[0, 1, 0, 1],
                    origin="lower",
                )

            ax.set_aspect(aspect="equal")

            # cbar = fig.colorbar(cs)
            # cbar.ax.set_ylabel("$f(x)$", rotation=270, labelpad=20)

            # Plot data, optimal actions, next query #################################
            ax.scatter(
                data_x[: n_initial_points - 1, 0],
                data_x[: n_initial_points - 1, 1],
                color="black",
            )
            ax.scatter(
                data_x[n_initial_points - 1 : algo_n_iterations, 0],
                data_x[n_initial_points - 1 : algo_n_iterations, 1],
                color=colors,
            )
            ax.scatter(bayes_action[..., 0], bayes_action[..., 1], color="green")

            # Splotlight cost
            if cost_fn == "r-spotlight":
                # previous_x = buffer["x"][algo_n_iterations - 1].squeeze()
                previous_x = bayes_action[0]
                previous_x = previous_x.cpu().detach().numpy()
                splotlight = plt.Rectangle(
                    (
                        previous_x[0] - parms.cost_func_hypers["radius"],
                        previous_x[1] - parms.cost_func_hypers["radius"],
                    ),
                    2 * parms.cost_func_hypers["radius"],
                    2 * parms.cost_func_hypers["radius"],
                    color="black",
                    linestyle="dashed",
                    fill=False,
                )
                ax.add_patch(splotlight)

    # fig.legend()
    if len(env_names) == 1 and len(algos) == 1:
        fig.suptitle(f"{env_names[0]} - {cost_function_names[cost_fn]}", fontsize=30)
        axs.set_ylabel(algos_name[0], rotation=90, fontsize=30)
        
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-3, 3), cmap="bwr"),
            ax=axs,
            shrink=0.9,
        )
        cb.ax.tick_params(labelsize=24)
        
    elif len(env_names) == 1:
        for ax, col in zip(axs, algos_name):
            ax.set_title(col, fontsize=30)
            
        axs[0].set_ylabel(env_names[0], rotation=90, fontsize=30)
        
        if env_names[0] == "NightLight":
            cb = fig.colorbar(
                mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-3, 3), cmap="gray"),
                ax=axs,
                shrink=0.9,
            )
        else:
            cb = fig.colorbar(
                mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-3, 3), cmap="bwr"),
                ax=axs[-1],
                shrink=0.9,
            )
        cb.ax.tick_params(labelsize=24)
        
    elif len(algos) == 1:
        for ax, row in zip(axs, env_names):
            ax.set_ylabel(row, rotation=90, fontsize=30)
        
        axs[0].set_title(algos_name[0], fontsize=30)
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-3, 3), cmap="bwr"),
            ax=axs[0],
            shrink=0.9,
        )
        cb.ax.tick_params(labelsize=24)
            
    else:
        for ax, col in zip(axs[0], algos_name):
            ax.set_title(col, fontsize=30)

        for ax, row in zip(axs[:, 0], env_names):
            ax.set_ylabel(row, rotation=90, fontsize=30)

        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-3, 3), cmap="bwr"),
            ax=axs[:, -1],
            shrink=0.9,
        )
        cb.ax.tick_params(labelsize=24)

    plt.savefig(f"plots/{args.setting}_{args.kernel + '_' if args.kernel != 'RBF' else ''}{en}_{cost_fn}_init{args.n_initial_points}_hidden{args.hidden_dim}_rs{args.n_restarts}.pdf")
    plt.savefig(f"plots/{args.setting}_{args.kernel + '_' if args.kernel != 'RBF' else ''}{en}_{cost_fn}_init{args.n_initial_points}_hidden{args.hidden_dim}_rs{args.n_restarts}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    for en in env_noises:
        for cost_fn in cost_functions:
            # for seed in seeds:
            seed = 0
            results = read_results(seed, en, cost_fn, args)
            draw_result(results, seed, en, cost_fn, args)
