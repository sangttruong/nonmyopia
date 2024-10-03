#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Evaluate and plot."""
from argparse import ArgumentParser

import gpytorch

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from main import Parameters

from tueplots import bundles
from utils import eval_func, make_env, set_seed, str2bool

plt.rcParams.update(bundles.neurips2024())


def eval_and_plot_2D_with_posterior(
    env,
    surr_model,
    parms,
    buffer,
    iteration,
    next_x=None,
    n_space=200,
    embedder=None,
    *args,
    **kwargs,
):
    r"""Evaluate and plot 2D function."""
    real_loss, bayes_action = eval_func(
        env, surr_model, parms, buffer, iteration, embedder
    )

    data_x = buffer["x"].cpu().detach()
    if next_x:
        next_x = next_x.cpu().detach()

    # Plotting ###############################################################
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    if embedder is not None:
        bounds_plot_x = bounds_plot_y = [0, n_space - 1]
    else:
        bounds_plot_x, bounds_plot_y = np.array([0, 1]), np.array([0, 1])
    ax[0].set(xlabel="$x_1$", ylabel="$x_2$", xlim=bounds_plot_x, ylim=bounds_plot_y)
    ax[1].set(xlabel="$x_1$", ylabel="$x_2$", xlim=bounds_plot_x, ylim=bounds_plot_y)

    if parms.algo == "HES":
        title = (
            "HES"
            + ("-TS" if parms.algo_ts else "")
            + ("-AM" if parms.amortized else "")
            + (f"-{parms.algo_lookahead_steps}")
        )
    else:
        title = parms.algo

    ax[0].set_title(label=title)
    ax[1].set_title(label="Posterior mean")

    # Plot function in 2D ####################################################
    X_domain, Y_domain = bounds_plot_x, bounds_plot_y
    X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
    X, Y = np.meshgrid(X, Y)
    XY = torch.tensor(np.array([X, Y]))  # >> 2 x 100 x 100
    Z = env(XY.to(parms.device, parms.torch_dtype).reshape(2, -1).T).reshape(X.shape)
    Z = Z.cpu().detach()

    Z_post = surr_model.posterior(
        XY.to(parms.device, parms.torch_dtype).permute(1, 2, 0)
    ).mean
    Z_post = Z_post.squeeze(-1).cpu().detach()
    # >> 100 x 100 x 1

    if embedder is not None:
        X, Y = (
            embedder.decode(XY.permute(1, 2, 0).to(device=embedder.device))
            .permute(2, 0, 1)
            .long()
            .cpu()
            .detach()
            .numpy()
        )

    cs = ax[0].contourf(X, Y, Z, levels=30, cmap="bwr", alpha=0.7)
    ax[0].set_aspect(aspect="equal")

    cs1 = ax[1].contourf(X, Y, Z_post, levels=30, cmap="bwr", alpha=0.7)
    ax[1].set_aspect(aspect="equal")

    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel("$f(x)$", rotation=270, labelpad=20)

    cbar1 = fig.colorbar(cs1)
    cbar1.ax.set_ylabel(r"$\hat{f}(x)$", rotation=270, labelpad=20)

    # Plot data, optimal actions, next query #################################
    if embedder is not None:
        ax[0].scatter(
            *embedder.decode(data_x[:iteration].to(device=embedder.device))
            .cpu()
            .long()
            .T,
            label="Data",
        )
        ax[0].scatter(
            *embedder.decode(bayes_action.to(device=embedder.device)).cpu().long().T,
            label="Action",
        )
        if next_x:
            ax[0].scatter(
                *embedder.decode(next_x.unsqueeze(0).to(device=embedder.device))
                .cpu()
                .long()
                .T,
                label="Next query",
            )

        # Draw grid
        plt.vlines(np.arange(0, n_space), 0, n_space, linestyles="dashed", alpha=0.05)
        plt.hlines(np.arange(0, n_space), 0, n_space, linestyles="dashed", alpha=0.05)

        # Splotlight cost
        previous_x = embedder.decode(
            buffer["x"][iteration - 1].squeeze().to(device=embedder.device)
        ).long()
        previous_x = previous_x.cpu().detach()
        splotlight = plt.Rectangle(
            (
                (
                    previous_x[0]
                    - parms.cost_func_hypers["radius"]
                    * n_space
                    / (parms.bounds[0, 1] - parms.bounds[0, 0])
                ).cpu(),
                (
                    previous_x[1]
                    - parms.cost_func_hypers["radius"]
                    * n_space
                    / (parms.bounds[1, 1] - parms.bounds[1, 0])
                ).cpu(),
            ),
            (
                2
                * parms.cost_func_hypers["radius"]
                * n_space
                / (parms.bounds[0, 1] - parms.bounds[0, 0])
            ).cpu(),
            (
                2
                * parms.cost_func_hypers["radius"]
                * n_space
                / (parms.bounds[1, 1] - parms.bounds[1, 0])
            ).cpu(),
            color="black",
            linestyle="dashed",
            fill=False,
        )
        ax[0].add_patch(splotlight)

        ax[0].set_xticks(range(0, n_space, 2))
        ax[0].set_yticks(range(0, n_space, 2))

    else:
        ax[0].scatter(data_x[:iteration, 0], data_x[:iteration, 1], label="Data")
        ax[0].scatter(bayes_action[..., 0], bayes_action[..., 1], label="Action")

        if "actions" in kwargs and kwargs["actions"] is not None:
            actions = kwargs["actions"]
            actions = actions.cpu().detach().numpy()
            ax[0].scatter(
                actions[..., 0].reshape(-1, 1),
                actions[..., 1].reshape(-1, 1),
                label="Imaged Action",
            )
        if "X" in kwargs:
            for X in kwargs["X"].cpu().numpy()[::-1]:
                ax[0].scatter(X[..., 0].reshape(-1, 1), X[..., 1].reshape(-1, 1))

        if next_x:
            ax[0].scatter(next_x[..., 0], next_x[..., 1], label="Next query")

        # Splotlight cost
        previous_x = buffer["x"][iteration - 1].squeeze()
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
        ax[0].add_patch(splotlight)

    fig.legend()

    plt.savefig(f"{parms.save_dir}/{iteration}.pdf")
    plt.close()

    return real_loss, bayes_action


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--task", type=str, default="topk")
    parser.add_argument("--env_name", type=str, default="SynGP")
    parser.add_argument("--env_noise", type=float, default=0.01)
    parser.add_argument("--env_discretized", type=str2bool, default=False)
    parser.add_argument("--algo", type=str, default="HES-TS-AM-20")
    parser.add_argument("--cost_fn", type=str, default="r-spotlight")
    parser.add_argument("--plot", type=str2bool, default=False)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--cont", type=str2bool, default=False)
    parser.add_argument("--iter", type=int, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    parms = Parameters(args)
    parms.algo_n_iterations = args.iter

    # Init environment
    env = make_env(
        name=parms.env_name,
        x_dim=parms.x_dim,
        bounds=parms.bounds,
        noise_std=parms.env_noise,
    )
    env = env.to(
        dtype=parms.torch_dtype,
        device=parms.device,
    )
    surr_file = f"results/{parms.env_name}_{parms.env_noise}{'_discretized' if parms.env_discretized else ''}/{args.algo}_{args.cost_fn}_seed{parms.seed}/surr_model_{parms.algo_n_iterations-1}.pt"
    buffer_file = f"results/{parms.env_name}_{parms.env_noise}{'_discretized' if parms.env_discretized else ''}/{args.algo}_{args.cost_fn}_seed{parms.seed}/buffer.pt"

    buffer = torch.load(buffer_file, map_location=parms.device).to(
        dtype=parms.torch_dtype
    )
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=gpytorch.priors.NormalPrior(0, 1e-2)
    )
    surr_model = SingleTaskGP(
        buffer["x"][: parms.algo_n_iterations + 1],
        buffer["y"][: parms.algo_n_iterations + 1],
        likelihood=likelihood,
    ).to(parms.device, dtype=parms.torch_dtype)

    mll = ExactMarginalLogLikelihood(surr_model.likelihood, surr_model)
    fit_gpytorch_model(mll)

    eval_and_plot_2D_with_posterior(
        env=env,
        surr_model=surr_model,
        parms=parms,
        buffer=buffer,
        iteration=parms.algo_n_iterations - 1,
    )
