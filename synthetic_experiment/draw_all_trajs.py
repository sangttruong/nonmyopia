#!/usr/bin/env python3


r"""Evaluate and plot."""
import pickle
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


algos_name = [
    "SR",
    "EI",
    "PI",
    "UCB",
    "KG",
    "HES-TS-20",
    "Ours",
]

algos = [
    "qSR",
    "qEI",
    "qPI",
    "qUCB",
    "qKG",
    "qMSL",
    "HES-TS-AM-20",
]

seeds = {
    "qSR": 2,
    "qEI": 3,
    "qPI": 5,
    "qUCB": 2,
    "qKG": 3,
    "HES-TS-20": 2,
    "HES-TS-AM-20": 2,
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

cost_function_names = {
    "euclidean": "Euclidean cost",
    "manhattan": "Manhattan cost",
    "r-spotlight": "$r$-spotlight cost",
    "non-markovian": "Non-Markovian cost",
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_results(seed, en, cost_fn):
    outputs = {}

    for env_name in env_names:
        for algo in algos:
            seed = seeds[algo]

            print(f"Reading {env_name} - {algo}")
            x_dim, bounds, radius, n_initial_points, algo_n_iterations = get_env_info(
                env_name, device
            )
            surr_file = f"results/{env_name}_{en}/{algo}_{cost_fn}_seed{seed}_init5/surr_model_{algo_n_iterations-1}.pt"
            buffer_file = (
                f"results/{env_name}_{en}/{algo}_{cost_fn}_seed{seed}_init5/buffer.pt"
            )

            buffer = torch.load(buffer_file, map_location=device).to(
                dtype=torch.float64
            )
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=gpytorch.priors.NormalPrior(0, 1e-2)
            )
            surr_model = SingleTaskGP(
                buffer["x"][: algo_n_iterations + 1],
                buffer["y"][: algo_n_iterations + 1],
                likelihood=likelihood,
            ).to(device, dtype=torch.float64)

            mll = ExactMarginalLogLikelihood(surr_model.likelihood, surr_model)
            fit_gpytorch_model(mll)

            outputs[(env_name, algo)] = [buffer, surr_model]

    return outputs


class PseudoArgs:
    def __init__(self):
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


def draw_result(results, seed, en, cost_fn):
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
        colors = [cmap(1 - norm(lhi)) for lhi in range(total_steps)]

        for algo in algos:
            seed = seeds[algo]

            buffer, surr_model = results[(env_name, algo)]
            row_idx = env_names.index(env_name)
            col_idx = algos.index(algo)
            ax = axs[row_idx][col_idx]

            # Parse args
            args = PseudoArgs()
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
            X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
            X, Y = np.meshgrid(X, Y)
            XY = torch.tensor(np.array([X, Y]))  # >> 2 x 100 x 100
            Z = env(XY.to(parms.device, parms.torch_dtype).reshape(2, -1).T).reshape(
                X.shape
            )
            Z = Z.cpu().detach()

            cs = ax.contourf(X, Y, Z, levels=30, cmap="bwr", alpha=0.7)
            ax.set_aspect(aspect="equal")

            # cbar = fig.colorbar(cs)
            # cbar.ax.set_ylabel("$f(x)$", rotation=270, labelpad=20)

            # Plot data, optimal actions, next query #################################
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

    plt.savefig(f"plots/{seed}_{en}_{cost_fn}.pdf")
    plt.close()


if __name__ == "__main__":
    for en in env_noises:
        for cost_fn in cost_functions:
            # for seed in seeds:
            seed = 0
            results = read_results(seed, en, cost_fn)
            draw_result(results, seed, en, cost_fn)
