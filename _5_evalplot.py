#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Evaluate and plot."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from tueplots import bundles

plt.rcParams.update(bundles.iclr2023())


def eval(cfg, acqf, func):
    # Quality of the best decision from the current posterior distribution ###
    # Initialize A consistently across fantasies
    A = torch.rand(
        [1, 1, cfg.n_actions, cfg.x_dim],
        requires_grad=True,
        device=cfg.device,
        dtype=cfg.torch_dtype,
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam([A], lr=cfg.acq_opt_lr)
    ba_l, ba_u = cfg.bounds

    for i in range(cfg.acq_opt_iter):
        A_ = A.permute(1, 0, 2, 3)
        A_ = torch.sigmoid(A_) * (ba_u - ba_l) + ba_l
        posterior = acqf.model.posterior(A_)
        fantasized_outcome = acqf.action_sampler(posterior)
        # >>> n_fantasy_at_action_pts x n_fantasy_at_design_pts
        # ... x batch_size x num_actions x 1

        fantasized_outcome = fantasized_outcome.squeeze(dim=-1)
        # >>> n_fantasy_at_action_pts x n_fantasy_at_design_pts
        # ... x batch_size x num_actions

        losses = -acqf.loss_function(A=A_, Y=fantasized_outcome)
        # >>> n_fantasy_at_design_pts x batch_size

        loss = losses.mean(dim=0).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 200 == 0:
            print(f"Eval optim round: {i}, Loss: {loss.item():.2f}")

    A = torch.sigmoid(A) * (ba_u - ba_l) + ba_l
    bayes_loss = acqf.loss_function(A=A, Y=fantasized_outcome)
    bayes_loss = bayes_loss[0, 0].cpu().detach().numpy()
    bayes_action = A[0, 0].cpu().detach().numpy()
    real_loss = acqf.loss_function(A=A, Y=func(A)[None, ...])[0, 0]
    return real_loss, bayes_action


def eval_and_plot_2D(func, cfg, acqf, next_x, buffer, iteration):
    r"""Evaluate and plot 2D function."""
    real_loss, bayes_action = eval(cfg, acqf, func)
    data_x = buffer["x"].cpu().detach().numpy()
    next_x = next_x.cpu().detach().numpy()

    # Plotting ###############################################################
    fig, ax = plt.subplots(1, 1)
    bounds_plot = cfg.bounds
    ax.set(xlabel="$x_1$", ylabel="$x_2$", xlim=bounds_plot, ylim=bounds_plot)
    title = "$\mathcal{H}_{\ell, \mathcal{A}}$-Entropy Search " + cfg.task
    ax.set_title(label=title)

    # Plot function in 2D ####################################################
    X_domain, Y_domain = cfg.bounds, cfg.bounds
    n_space = 200
    X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
    X, Y = np.meshgrid(X, Y)
    XY = torch.tensor(np.array([X, Y]))
    Z = func(XY.reshape(2, -1).T).reshape(X.shape)
    cs = ax.contourf(X, Y, Z, levels=30, cmap="bwr", alpha=0.7)
    ax.set_aspect(aspect="equal")
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel("$f(x)$", rotation=270, labelpad=20)

    # Plot data, optimal actions, next query #################################
    ax.scatter(data_x[:iteration, 0], data_x[:iteration, 1], label="Data")
    ax.scatter(bayes_action[:, 0], bayes_action[:, 1], label="Action")
    ax.scatter(next_x[..., 0], next_x[..., 1], label="Next query")
    plt.legend()
    plt.savefig(f"{cfg.save_dir}/{iteration}.pdf")
    plt.close()

    return real_loss


def eval_and_plot_1D(func, cfg, acqf, next_x, buffer, iteration):
    r"""Evaluate and plot 1D function."""
    real_loss, bayes_action = eval(cfg, acqf, func)
    data_x = buffer["x"].cpu().detach().numpy()
    data_y = buffer["y"].cpu().detach().numpy()
    next_x = next_x.cpu().detach().numpy()

    # Plotting ###############################################################
    fig, ax = plt.subplots(1, 1)
    bounds_plot = cfg.bounds
    y_bounds = (-5, 5)
    ax.set(xlabel="$x$", ylabel="$y$", xlim=bounds_plot, ylim=y_bounds)
    title = "$\mathcal{H}_{\ell, \mathcal{A}}$-Entropy Search " + cfg.task
    ax.set_title(label=title)

    # Plot function and posterior in 1D #######################################
    x = torch.linspace(-1, 1, 100).reshape(-1, 1)
    posterior = acqf.model.posterior(x)
    test_y = posterior.mean
    lower, upper = posterior.mvn.confidence_region()
    x = x.cpu().detach().numpy()
    test_y = test_y.cpu().detach().numpy()
    lower = lower.cpu().detach().numpy()
    upper = upper.cpu().detach().numpy()
    plt.plot(x, func(x).numpy(), alpha=0.2, label="Ground truth")
    plt.plot(x, test_y, label="Posterior mean")
    plt.fill_between(x, lower, upper, alpha=0.25)

    # Plot data, optimal actions, next query #################################
    plt.vlines(data_x[iteration], -5, 5, color="black", label="current location")
    plt.vlines(data_x[iteration] - 0.1, -5, 5, color="black", linestyle="--")
    plt.vlines(data_x[iteration] + 0.1, -5, 5, color="black", linestyle="--")
    plt.vlines(next_x, -5, 5, color="red", label="optimal query")
    plt.vlines(bayes_action, -5, 5, color="blue", label="Bayes action")

    # Scatter plot of all points using buffer.x and buffer.y with
    # gradient color from red to blue indicating the order of point in list
    plt.scatter(
        data_x[:iteration],
        data_y[:iteration],
        c=range(iteration),
        cmap="Reds",
        marker="*",
        zorder=99,
    )
    plt.savefig(f"{cfg.save_dir}/posterior_{iteration}.pdf")
    plt.close()

    return real_loss


def eval_and_plot(config, env, acqf, buffer, next_x, iteration):
    r"""Draw the posterior of the model."""
    if config.x_dim == 1:
        eval_and_plot_1D(config, env, acqf, buffer, next_x, iteration)
    elif config.x_dim == 2:
        eval_and_plot_2D(config, env, acqf, buffer, next_x, iteration)
    else:
        raise NotImplementedError


def draw_metric(save_dir, metrics, algos):
    r"""Draw the evaluation metric of each algorithm."""
    if isinstance(metrics, list):
        metrics = np.array(metrics)

    plt.figure()
    for i, algo in enumerate(algos):
        mean = np.mean(metrics[i], axis=0)
        lower = np.min(metrics[i], axis=0)
        upper = np.max(metrics[i], axis=0)
        x = list(range(1, mean.shape[0] + 1))
        plt.plot(x, mean, label=algo)
        plt.fill_between(x, lower, upper, alpha=0.25)

    plt.xlabel("Iteration")
    plt.ylabel("Eval metric")
    plt.savefig(f"{save_dir}/eval_metric.pdf")
    plt.close()
