#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Evaluate and plot."""
import os
import pickle
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import gpytorch

import matplotlib.cm as cm
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import numpy as np
import torch
from actor import Actor
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from main import Parameters

from tqdm import tqdm
from tueplots import bundles, figsizes
from utils import create_gif, make_env, set_seed, str2bool

plt.rcParams.update(bundles.iclr2024())
gif_size = figsizes.iclr2024(nrows=1, ncols=3)["figure.figsize"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_acquisition_value(
    parms,
    buffer,
    actor,
    iteration,
    surr_model_state_dict,
    x1,
    f_posterior,
):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=gpytorch.priors.NormalPrior(0, 1e-2)
    )

    surr_model = SingleTaskGP(
        buffer["x"][:iteration],
        buffer["y"][:iteration],
        likelihood=likelihood,
        covar_module=parms.kernel,
    ).to(device, dtype=parms.torch_dtype)
    mll = ExactMarginalLogLikelihood(surr_model.likelihood, surr_model)
    fit_gpytorch_model(mll)
    surr_model.load_state_dict(surr_model_state_dict)

    # Construct acqf
    actor.construct_acqf(surr_model=surr_model, buffer=buffer[:iteration])
    actor.reset_parameters(
        buffer=buffer[:iteration],
        bo_iter=iteration - parms.n_initial_points,
        embedder=None,
        prev_chosen_idx=buffer["chosen_idx"][iteration - 1],
    )

    reshaped_x1 = x1[:, None].expand(-1, 64, -1).reshape(4096, -1).to(actor.maps[0])
    actor.maps[0] = reshaped_x1.requires_grad_(False)
    actor._parameters = actor.maps[1:]

    # Query and observe next point
    output = actor.query(
        buffer=buffer,
        iteration=iteration,
        embedder=None,
        f_posterior=f_posterior,
        save_trajectory=False,
    )
    acqf_value = -output["all_losses"]  # + output["all_costs"]
    acqf_value = acqf_value.reshape(64, 64).max(dim=-1).values

    return acqf_value


def eval_and_plot_2D_with_posterior(
    actor,
    env,
    traj,
    chosen_idx,
    f_posterior,
    surr_model_state_dict,
    parms,
    buffer,
    iteration,
    n_space=200,
    embedder=None,
    disable_acquisition=False,
    *args,
    **kwargs,
):
    r"""Evaluate and plot 2D function."""
    n_epochs = len(traj)
    if n_epochs < 10:
        plotting_epochs = list(range(0, n_epochs, 1))
    else:
        plotting_epochs = list(range(0, n_epochs, n_epochs // 10))
        if n_epochs % 10 != 0:
            plotting_epochs.append(n_epochs - 1)

    # Plotting ###############################################################
    bounds_plot_x, bounds_plot_y = np.array([0, 1]), np.array([0, 1])

    # Plot function in 2D ####################################################
    X_domain, Y_domain = bounds_plot_x, bounds_plot_y
    X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
    X, Y = np.meshgrid(X, Y)
    XY = torch.tensor(np.array([X, Y]))  # >> 2 x 100 x 100
    Z = env(XY.to(parms.device, parms.torch_dtype).reshape(2, -1).T).reshape(X.shape)
    Z = Z.cpu().detach()

    Z_posterior = f_posterior(
        XY.to(parms.device, parms.torch_dtype).reshape(2, -1).T
    ).reshape(X.shape)
    Z_posterior = Z_posterior.cpu().detach()

    ##### SAVE POSTERIOR PLOT #####
    fig, axes = plt.subplots(
        ncols=3 - int(disable_acquisition),
        nrows=1,
        figsize=(1.25 * gif_size[0], 2 * gif_size[1]),
    )
    # Set label and bounds
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")
    axes[1].set_xlim(*bounds_plot_x)
    axes[1].set_ylim(*bounds_plot_y)

    # Scattering all points in buffer except the initial points
    axes[1].contourf(X, Y, Z, levels=30, cmap="bwr", alpha=0.7)
    axes[1].set_aspect(aspect="equal")
    axes[1].set_title("Ground Truth")

    ##### SAVE ACQUISITION #####
    # For spotlight cost only
    # The last point of the buffer is buffer["x"][iteration - 1]
    # Create a meshgrid of 8 x 8 points centered around the last point with radius parms.radius
    # The meshgrid is then reshaped to 64 x 2

    if not disable_acquisition:
        rX = torch.linspace(
            max(buffer["x"][iteration - 1, 0] - parms.radius, 0),
            min(buffer["x"][iteration - 1, 0] + parms.radius, 1),
            8,
        )
        rY = torch.linspace(
            max(buffer["x"][iteration - 1, 1] - parms.radius, 0),
            min(buffer["x"][iteration - 1, 1] + parms.radius, 1),
            8,
        )
        rX, rY = torch.meshgrid(rX, rY)
        rXY = torch.stack([rX, rY], dim=-1).reshape(-1, 2)

        parms.n_restarts *= 64
        A = compute_acquisition_value(
            parms=parms,
            buffer=buffer,
            actor=actor,
            iteration=iteration,
            surr_model_state_dict=surr_model_state_dict,
            x1=rXY,
            f_posterior=f_posterior,
        )
        parms.n_restarts = int(parms.n_restarts / 64)
        A = A.reshape(rX.shape)
        A = A.cpu().detach()

        # Set label and bounds
        axes[2].set_xlabel("$x_1$")
        axes[2].set_ylabel("$x_2$")
        axes[2].set_xlim(*bounds_plot_x)
        axes[2].set_ylim(*bounds_plot_y)

        # Scattering all points in buffer except the initial points
        levels = np.linspace(Z_posterior.min(), Z_posterior.max(), 30)
        axes[2].contourf(rX, rY, A, levels=levels, cmap="bwr", alpha=0.7)
        axes[2].set_aspect(aspect="equal")
        axes[2].set_title("Acquisition Value")

        splotlight = plt.Rectangle(
            (
                buffer["x"][iteration - 1, 0].cpu() - parms.radius,
                buffer["x"][iteration - 1, 1].cpu() - parms.radius,
            ),
            2 * parms.radius,
            2 * parms.radius,
            color="black",
            linestyle="dashed",
            fill=False,
        )
        axes[2].add_patch(splotlight)

    ##### SAVE TRAJECTORY PLOT #####
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

    norm = mcolors.Normalize(
        vmin=parms.n_initial_points - 1, vmax=parms.algo_n_iterations
    )
    cmap = cm.viridis  # Choose a colormap
    list_saved_files = []

    print(f"Drawing iter#{iteration}")
    for npe, pe in enumerate(tqdm(plotting_epochs)):

        # plt.figure(figsize=(2*gif_size[0]/3, 2*gif_size[1]))
        axes[0].cla()
        # Set label and bounds
        axes[0].set_xlabel("$x_1$")
        axes[0].set_ylabel("$x_2$")
        axes[0].set_xlim(*bounds_plot_x)
        axes[0].set_ylim(*bounds_plot_y)

        # Scattering all points in buffer except the initial points
        axes[0].contourf(X, Y, Z_posterior, levels=30, cmap="bwr", alpha=0.7)
        axes[0].set_aspect(aspect="equal")

        for i in range(parms.n_initial_points - 1, iteration - 1):
            axes[0].plot(
                buffer["x"][i : i + 2, 0].cpu().numpy(),
                buffer["x"][i : i + 2, 1].cpu().numpy(),
                color=cmap(norm(i)),
            )
        axes[0].scatter(
            buffer["x"][parms.n_initial_points - 1 : iteration, 0].cpu().numpy(),
            buffer["x"][parms.n_initial_points - 1 : iteration, 1].cpu().numpy(),
            color=[
                cmap(norm(bi)) for bi in range(parms.n_initial_points - 1, iteration)
            ],
        )

        if parms.algo == "HES":
            title = (
                "HES"
                + ("-TS" if parms.algo_ts else "")
                + ("-AM" if parms.amortized else "")
                + (f"-{parms.algo_lookahead_steps}")
            )
            if title == "HES-TS-20":
                title = "MSL"
        else:
            title = parms.algo
        title += f" BO\#{iteration - parms.n_initial_points}"

        list_restarts = list(range(parms.n_restarts))
        list_restarts.remove(chosen_idx)
        list_restarts = list_restarts + [chosen_idx]

        for nr in list_restarts:
            iter_traj = traj[pe]
            list_traj = [np.array(x) for x in iter_traj]

            # Loop through trajectories and draw them
            for batch_idx, batch in enumerate(list_traj):
                desired_shape = nf_design_pts[:batch_idx]
                batch = batch.reshape(*desired_shape, parms.n_restarts, 2)

                x = batch[..., 0]
                y = batch[..., 1]
                if nr == chosen_idx:
                    color = cmap(norm(batch_idx + iteration - 1))
                    opacity = 1
                else:
                    color = "grey"
                    opacity = 0.1

                if batch_idx == 0:  # For the first batch, just plot points
                    axes[0].plot(
                        [buffer["x"][iteration - 1, 0].cpu().numpy(), x[nr]],
                        [buffer["x"][iteration - 1, 1].cpu().numpy(), y[nr]],
                        color=color,
                        alpha=opacity,
                    )
                    axes[0].scatter(x[nr], y[nr], color=color, alpha=opacity)
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
                            axes[0].plot(x, y, color=color, alpha=opacity)
                            axes[0].scatter(x[1], y[1], color=color, alpha=opacity)
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
                                axes[0].plot(x, y, color=color, alpha=opacity)
                                axes[0].scatter(x[1], y[1], color=color, alpha=opacity)

        # cbar = plt.colorbar(cs)
        # cbar.ax.set_ylabel("$f(x)$", rotation=270, labelpad=20)

        # plt.colorbar(
        #     plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        #     label="Concentration",
        #     ax=plt.gca(),
        # )
        axes[0].set_title(title)
        plt.savefig(f"{parms.save_dir}/traj_{iteration:03d}_{npe:02d}.png", dpi=300)
        list_saved_files.append(f"{parms.save_dir}/traj_{iteration:03d}_{npe:02d}.png")

    plt.close()
    return list_saved_files


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--task", type=str, default="topk")
    parser.add_argument("--env_name", type=str, default="SynGP")
    parser.add_argument("--env_noise", type=float, default=0.0)
    parser.add_argument("--env_discretized", type=str2bool, default=False)
    parser.add_argument("--algo", type=str, default="HES-TS-AM-20")
    parser.add_argument("--cost_fn", type=str, default="r-spotlight")
    parser.add_argument("--n_initial_points", type=int, default=-1)
    parser.add_argument("--n_restarts", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--kernel", type=str, default="RBF")
    parser.add_argument("--plot", type=str2bool, default=False)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--cont", type=str2bool, default=False)
    parser.add_argument("--iter_start", type=int)
    parser.add_argument("--iter_end", type=int)
    parser.add_argument("--gif_only", type=str2bool, default=False)
    parser.add_argument("--disable_acquisition", type=str2bool, default=False)
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--max_workers", type=int, default=8)

    args = parser.parse_args()

    set_seed(args.seed)
    parms = Parameters(args)

    # Load trajectory
    if not args.gif_only:
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
        buffer_file = os.path.join(parms.save_dir, "buffer.pt")

        buffer = torch.load(buffer_file, map_location=parms.device).to(
            dtype=parms.torch_dtype
        )

        list_saved_files = []

        if args.algo == "HES-TS-AM-20":
            parms.algo == "HES-TS-20"
            parms.amortized = False

        parms.n_restarts *= 64
        actor = Actor(parms=parms)
        parms.algo = args.algo
        parms.n_restarts = int(parms.n_restarts / 64)

        for traj_iter in range(parms.n_initial_points, parms.algo_n_iterations):
            traj_file = os.path.join(parms.save_dir, f"trajectory_{traj_iter}.pkl")
            traj, chosen_traj_idx = pickle.load(open(traj_file, "rb"))

            post_file = os.path.join(
                parms.save_dir, f"posterior_sample_{traj_iter}.pkl"
            )
            f_posterior = pickle.load(open(post_file, "rb"))

            surr_model_file = os.path.join(parms.save_dir, f"surr_model_{traj_iter}.pt")
            surr_model_state_dict = torch.load(
                surr_model_file, map_location=parms.device
            )

            # Adjust lookahead steps
            if actor.algo_lookahead_steps > 1 and (
                parms.algo_n_iterations - traj_iter < actor.algo_lookahead_steps
            ):
                actor.algo_lookahead_steps = parms.algo_n_iterations - traj_iter

            saved_files = eval_and_plot_2D_with_posterior(
                actor=actor,
                env=env,
                traj=traj,
                chosen_idx=chosen_traj_idx,
                f_posterior=f_posterior,
                surr_model_state_dict=surr_model_state_dict,
                parms=parms,
                buffer=buffer,
                iteration=traj_iter,
                disable_acquisition=args.disable_acquisition,
            )
            list_saved_files.extend(saved_files)

        # Sort files by iteration
        list_saved_files = sorted(list_saved_files)

    else:
        list_saved_files = os.listdir(parms.save_dir)
        # Filter out files that are not trajectory files
        list_saved_files = [
            f for f in list_saved_files if f.startswith("traj") and f.endswith(".png")
        ]

        # Sort files by iteration
        list_saved_files = sorted(list_saved_files)

        # Add full path
        list_saved_files = [os.path.join(parms.save_dir, f) for f in list_saved_files]

    create_gif(list_saved_files, os.path.join(parms.save_dir, "trajectory.gif"))
