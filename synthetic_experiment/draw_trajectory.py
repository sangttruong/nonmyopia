#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Evaluate and plot."""
import pickle
from argparse import ArgumentParser

import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
    traj,
    chosen_idx,
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
    data_x = buffer["x"].cpu().detach()
    n_epochs = len(traj)
    plotting_epochs = list(range(0, n_epochs, 50))

    # Plotting ###############################################################
    fig, axs = plt.subplots(
        parms.n_restarts,
        len(plotting_epochs),
        figsize=(4 * len(plotting_epochs), 4 * parms.n_restarts),
    )
    bounds_plot_x, bounds_plot_y = np.array([0, 1]), np.array([0, 1])
    for nr in range(parms.n_restarts):
        for npe, pe in enumerate(plotting_epochs):
            axs[nr][npe].set(
                xlabel="$x_1$", ylabel="$x_2$", xlim=bounds_plot_x, ylim=bounds_plot_y
            )

    if parms.algo == "HES":
        title = (
            "HES"
            + ("-TS" if parms.algo_ts else "")
            + ("-AM" if parms.amortized else "")
            + (f"-{parms.algo_lookahead_steps}")
        )
    else:
        title = parms.algo
    title += f" BO\#{iteration - parms.n_initial_points + 1}"

    # Plot function in 2D ####################################################
    X_domain, Y_domain = bounds_plot_x, bounds_plot_y
    X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
    X, Y = np.meshgrid(X, Y)
    XY = torch.tensor(np.array([X, Y]))  # >> 2 x 100 x 100
    Z = env(XY.to(parms.device, parms.torch_dtype).reshape(2, -1).T).reshape(X.shape)
    Z = Z.cpu().detach()

    if parms.algo_ts:
        nf_design_pts = [1] * parms.algo_lookahead_steps
    else:
        if parms.algo_lookahead_steps == 0:
            nf_design_pts = []
        elif parms.algo_lookahead_steps == 1:
            nf_design_pts = [64]
        elif parms.algo_lookahead_steps == 2:
            nf_design_pts = [64, 8]  # [64, 64]
        elif parms.algo_lookahead_steps == 3:
            nf_design_pts = [64, 4, 2]  # [64, 32, 8]
        elif parms.algo_lookahead_steps >= 4:
            nf_design_pts = [64, 4, 2, 1]  # [16, 8, 8, 8]
            nf_design_pts = nf_design_pts + [1] * (parms.algo_lookahead_steps - 4)

    norm = mcolors.Normalize(vmin=0, vmax=parms.algo_lookahead_steps)
    cmap = cm.viridis  # Choose a colormap

    for nr in range(parms.n_restarts):
        for npe, pe in enumerate(plotting_epochs):
            print(f"Drawing restart#{nr} epoch#{pe}")
            ax = axs[nr][npe]

            cs = ax.contourf(X, Y, Z, levels=30, cmap="bwr", alpha=0.7)
            ax.set_aspect(aspect="equal")
            if nr == chosen_idx:
                title_color = "red"
            else:
                title_color = "black"
            ax.set_title(f"Restart {nr} - Epoch {pe}", color=title_color)

            iter_traj = traj[pe]
            list_traj = [np.array(x) for x in iter_traj]

            # Loop through trajectories and draw them
            for batch_idx, batch in enumerate(list_traj):
                desired_shape = nf_design_pts[:batch_idx]
                batch = batch.reshape(*desired_shape, parms.n_restarts, 2)
                x = batch[..., 0]
                y = batch[..., 1]
                color = cmap(norm(batch_idx))

                if batch_idx == 0:  # For the first batch, just plot points
                    ax.scatter(x[nr], y[nr], color=color)
                elif batch_idx == 1:
                    for i in range(batch.shape[0]):
                        tail = list_traj[batch_idx - 1][nr, :].reshape(-1, parms.x_dim)
                        head = batch[i, nr, :].reshape(-1, parms.x_dim)
                        for (
                            tp,
                            hp,
                        ) in zip(tail, head):
                            x = [tp[0], hp[0]]
                            y = [tp[1], hp[1]]
                            ax.plot(x, y, color=color)
                            ax.scatter(x[1], y[1], color=color)
                else:
                    # For next batches, connect points from previous batch
                    prev_batch = list_traj[batch_idx - 1].reshape(
                        *nf_design_pts[: batch_idx - 1], parms.n_restarts, 2
                    )
                    prev_batch_size = batch.shape[1]

                    for i in range(batch.shape[0]):
                        for j in range(prev_batch_size):
                            tail = prev_batch[j, ..., nr, :].reshape(-1, parms.x_dim)
                            head = batch[i, j, ..., nr, :].reshape(-1, parms.x_dim)
                            for (
                                tp,
                                hp,
                            ) in zip(tail, head):
                                x = [tp[0], hp[0]]
                                y = [tp[1], hp[1]]
                                ax.plot(x, y, color=color)
                                ax.scatter(x[1], y[1], color=color)

    # cbar = plt.colorbar(cs)
    # cbar.ax.set_ylabel("$f(x)$", rotation=270, labelpad=20)

    # plt.colorbar(
    #     plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    #     label="Concentration",
    #     ax=plt.gca(),
    # )
    # plt.title(title)
    plt.savefig(f"{parms.save_dir}/traj_{iteration}.png", dpi=200)
    plt.close()


def draw_loss_and_cost(parms, losses, costs, iteration, chosen_idx):
    # fig, axs = plt.subplots(1, parms.n_restarts, figsize=(5*parms.n_restarts, 4))
    fig = plt.figure(figsize=(5, 4))
    for ridx in range(parms.n_restarts):
        loss = [x[ridx] for x in losses]
        cost = [x[ridx] for x in costs]
        plt.plot(loss, alpha=0.1, color="blue")
        plt.plot(cost, alpha=0.1, color="orange")

    mean_loss = np.array(losses).mean(axis=1)
    mean_cost = np.array(costs).mean(axis=1)
    plt.plot(mean_loss, color="blue", label="Loss")
    plt.plot(mean_cost, color="orange", label="Cost")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"{parms.save_dir}/lossncost_{iteration}.png", dpi=200)
    plt.close()


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
    parser.add_argument("--traj_iter", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    parms = Parameters(args)
    parms.algo_n_iterations = args.traj_iter

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
    traj_file = f"results/{parms.env_name}_{parms.env_noise}{'_discretized' if parms.env_discretized else ''}/{args.algo}_{args.cost_fn}_seed{parms.seed}/trajectory_{args.traj_iter}.pkl"
    lossncost_file = f"results/{parms.env_name}_{parms.env_noise}{'_discretized' if parms.env_discretized else ''}/{args.algo}_{args.cost_fn}_seed{parms.seed}/lossncost_{args.traj_iter}.pkl"

    buffer = torch.load(buffer_file, map_location=parms.device).to(
        dtype=parms.torch_dtype
    )
    surr_model = SingleTaskGP(
        buffer["x"][: parms.algo_n_iterations + 1],
        buffer["y"][: parms.algo_n_iterations + 1],
        # input_transform=Normalize(
        #     d=parms.x_dim, bounds=parms.bounds.T),
    ).to(parms.device, dtype=parms.torch_dtype)

    mll = ExactMarginalLogLikelihood(surr_model.likelihood, surr_model)
    fit_gpytorch_model(mll)

    # Load trajectory
    traj, chosen_traj_idx = pickle.load(open(traj_file, "rb"))

    losses, costs = pickle.load(open(lossncost_file, "rb"))
    draw_loss_and_cost(
        parms, losses, costs, parms.algo_n_iterations - 1, chosen_traj_idx
    )

    eval_and_plot_2D_with_posterior(
        env=env,
        traj=traj,
        chosen_idx=chosen_traj_idx,
        surr_model=surr_model,
        parms=parms,
        buffer=buffer,
        iteration=parms.algo_n_iterations - 1,
    )
