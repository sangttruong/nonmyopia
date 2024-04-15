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
from botorch.optim.optimize import optimize_acqf, optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from _4_qhes import qLossFunctionTopK, qCostFunction

plt.rcParams.update(bundles.iclr2023())


def eval_func(
    cfg, acqf, func, buffer, optimal_value, iteration, embedder=None, *args, **kwargs
):
    # Quality of the best decision from the current posterior distribution ###

    if cfg.algo == "HES":
        # Initialize A consistently across fantasies
        A = torch.empty(
            [cfg.n_restarts, cfg.n_actions, cfg.x_dim],
            device=cfg.device,
            dtype=cfg.torch_dtype,
        )
        A = buffer["x"][iteration].clone().repeat(
            cfg.n_restarts, cfg.n_actions, 1)
        A = A + torch.randn_like(A) * 0.01
        A.requires_grad = True

        # Initialize optimizer
        optimizer = torch.optim.Adam([A], lr=0.01)

        for i in range(2000):
            posterior = acqf.model.posterior(A)
            fantasized_outcome = acqf.action_sampler(posterior)
            # >>> n_fantasy_at_action_pts x n_fantasy_at_design_pts
            # ... x batch_size x num_actions x 1

            fantasized_outcome = fantasized_outcome.squeeze(dim=-1)
            # >>> n_fantasy_at_action_pts x n_fantasy_at_design_pts
            # ... x batch_size x num_actions

            losses = acqf.loss_function(A=A, Y=fantasized_outcome) + acqf.cost_function(
                prev_X=buffer["x"][iteration].expand_as(A), current_X=A, previous_cost=0
            )
            # >>> n_fantasy_at_design_pts x batch_size

            loss = losses.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 200 == 0:
                print(f"Eval optim round: {i}, Loss: {loss.item():.2f}")

        A = A.cpu().detach()
        bayes_loss = acqf.loss_function(A=A, Y=fantasized_outcome)

        chosen_idx = torch.argmin(bayes_loss)
        bayes_loss = bayes_loss[chosen_idx].item()
        bayes_action = A[chosen_idx]

        real_loss = optimal_value + acqf.loss_function(
            A=bayes_action[None, ...], Y=func(bayes_action)[None, None, ...]
        )

    else:
        q = 1 + sum([cfg.n_samples**s for s in range(1,
                    cfg.algo_lookahead_steps + 1)])
        if cfg.env_name == "AntBO":
            choices = torch.tensor(
                list(range(20)), dtype=torch.long, device=cfg.device)
            choices = choices.reshape(-1, 1).expand(-1, cfg.x_dim)

            # Optimize acqf
            bayes_action, bayes_loss = optimize_acqf_discrete(
                acq_function=acqf,
                q=q,
                choices=choices,
            )
            real_loss = -func(bayes_action)
            bayes_action = bayes_action.cpu().detach().numpy()

        else:
            # bounds = torch.tensor(
            #     [cfg.bounds] * cfg.x_dim, dtype=cfg.torch_dtype, device=cfg.device
            # ).T

            # if cfg.algo == "qMSL":
            #     # Reinit 1-step model
            #     q = 1 + 64
            #     model = acqf.model
            #     acqf = qMultiStepLookahead(
            #         model=model,
            #         batch_sizes=[1],
            #         num_fantasies=[64],
            #     )

            # bayes_action, bayes_loss = optimize_acqf(
            #     acq_function=acqf,
            #     bounds=bounds,
            #     q=q,
            #     num_restarts=cfg.n_restarts,
            #     raw_samples=cfg.n_restarts,
            # )

            A = torch.empty(
                [1, cfg.n_restarts, cfg.n_actions, cfg.x_dim],
                device=cfg.device,
                dtype=cfg.torch_dtype,
            )
            A[0] = (
                buffer["x"][iteration].clone().repeat(
                    cfg.n_restarts, cfg.n_actions, 1)
            )
            A = A + torch.randn_like(A) * 0.01
            A.requires_grad = True

            # Initialize optimizer
            optimizer = torch.optim.Adam([A], lr=0.01)
            loss_function = qLossFunctionTopK(**cfg.loss_func_hypers)
            cost_function = qCostFunction(**cfg.cost_func_hypers)
            sampler = SobolQMCNormalSampler(
                sample_shape=64, resample=False, collapse_batch_dims=True
            )

            for i in range(2000):
                A_ = A.permute(1, 0, 2, 3)
                posterior = acqf.model.posterior(A_)
                fantasized_outcome = sampler(posterior)
                # >>> n_fantasy_at_action_pts x n_fantasy_at_design_pts
                # ... x batch_size x num_actions x 1

                fantasized_outcome = fantasized_outcome.squeeze(dim=-1)
                # >>> n_fantasy_at_action_pts x n_fantasy_at_design_pts
                # ... x batch_size x num_actions

                losses = loss_function(A=A_, Y=fantasized_outcome) + cost_function(
                    prev_X=buffer["x"][iteration], current_X=A_, previous_cost=0
                )
                # >>> n_fantasy_at_design_pts x batch_size

                loss = losses.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if i % 200 == 0:
                    print(f"Eval optim round: {i}, Loss: {loss.item():.2f}")

            A = A.permute(1, 0, 2, 3)
            A = A.cpu().detach()
            bayes_loss = loss_function(A=A, Y=fantasized_outcome)

            chosen_idx = torch.argmin(bayes_loss)
            bayes_loss = bayes_loss[chosen_idx].item()
            bayes_action = A[chosen_idx, 0]

            real_loss = optimal_value + loss_function(
                A=bayes_action[None, ...], Y=func(
                    bayes_action)[None, None, ...]
            )

    return real_loss, bayes_action


def eval_and_plot_2D(
    func,
    wm,
    cfg,
    acqf,
    next_x,
    buffer,
    optimal_value,
    iteration,
    n_space=100,
    embedder=None,
    *args,
    **kwargs,
):
    if iteration < cfg.n_initial_points:
        return eval_and_plot_2D_with_posterior(
            func,
            wm,
            cfg,
            acqf,
            next_x,
            buffer,
            optimal_value,
            iteration,
            n_space,
            embedder,
            *args,
            **kwargs,
        )
    else:
        return eval_and_plot_2D_without_posterior(
            func,
            wm,
            cfg,
            acqf,
            next_x,
            buffer,
            optimal_value,
            iteration,
            n_space,
            embedder,
            *args,
            **kwargs,
        )


def eval_and_plot_2D_with_posterior(
    func,
    wm,
    cfg,
    acqf,
    next_x,
    buffer,
    optimal_value,
    iteration,
    n_space=100,
    embedder=None,
    *args,
    **kwargs,
):
    r"""Evaluate and plot 2D function."""
    real_loss, bayes_action = eval_func(
        cfg, acqf, func, buffer, optimal_value, iteration
    )

    data_x = buffer["x"].cpu().detach()
    next_x = next_x.cpu().detach()

    # Plotting ###############################################################
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    if embedder is not None:
        bounds_plot_x = bounds_plot_y = [0, n_space - 1]
    else:
        bounds_plot_x, bounds_plot_y = cfg.bounds
    ax[0].set(xlabel="$x_1$", ylabel="$x_2$",
              xlim=bounds_plot_x, ylim=bounds_plot_y)
    ax[1].set(xlabel="$x_1$", ylabel="$x_2$",
              xlim=bounds_plot_x, ylim=bounds_plot_y)

    if cfg.algo == "HES":
        title = "$\mathcal{H}_{\ell, \mathcal{A}}$-Entropy Search " + cfg.task
    elif cfg.algo == "qMSL":
        title = "Multi-Step Trees " + cfg.task
    else:
        title = cfg.algo + " " + cfg.task

    ax[0].set_title(label=title)
    ax[1].set_title(label="Posterior mean")

    # Plot function in 2D ####################################################
    X_domain, Y_domain = cfg.bounds
    X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
    X, Y = np.meshgrid(X, Y)
    XY = torch.tensor(np.array([X, Y]))  # >> 2 x 100 x 100
    Z = func(XY.reshape(2, -1).T).reshape(X.shape)

    Z_post = wm.posterior(
        XY.to(cfg.device, cfg.torch_dtype).permute(1, 2, 0)).mean
    Z_post = Z_post.squeeze(-1).cpu().detach()
    # >> 100 x 100 x 1

    if embedder is not None:
        X, Y = (
            embedder.decode(XY.permute(1, 2, 0))
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
    cbar1.ax.set_ylabel("$\hat{f}(x)$", rotation=270, labelpad=20)

    # Plot data, optimal actions, next query #################################
    if embedder is not None:
        ax[0].scatter(
            *embedder.decode(data_x[:iteration]).long().T, label="Data")
        ax[0].scatter(*embedder.decode(bayes_action).long().T, label="Action")
        ax[0].scatter(
            *embedder.decode(next_x.unsqueeze(0)).long().T, label="Next query"
        )

        # Draw grid
        plt.vlines(np.arange(0, n_space), 0, n_space,
                   linestyles="dashed", alpha=0.05)
        plt.hlines(np.arange(0, n_space), 0, n_space,
                   linestyles="dashed", alpha=0.05)

        # Splotlight cost
        previous_x = embedder.decode(
            buffer["x"][iteration - 1].squeeze()).long()
        previous_x = previous_x.cpu().detach()
        splotlight = plt.Rectangle(
            (
                previous_x[0]
                - cfg.cost_func_hypers["radius"]
                * n_space
                / (cfg.bounds[0, 1] - cfg.bounds[0, 0]),
                previous_x[1]
                - cfg.cost_func_hypers["radius"]
                * n_space
                / (cfg.bounds[1, 1] - cfg.bounds[1, 0]),
            ),
            2
            * cfg.cost_func_hypers["radius"]
            * n_space
            / (cfg.bounds[0, 1] - cfg.bounds[0, 0]),
            2
            * cfg.cost_func_hypers["radius"]
            * n_space
            / (cfg.bounds[1, 1] - cfg.bounds[1, 0]),
            color="black",
            linestyle="dashed",
            fill=False,
        )
        ax[0].add_patch(splotlight)

        ax[0].set_xticks(range(0, n_space, 2))
        ax[0].set_yticks(range(0, n_space, 2))

    else:
        ax[0].scatter(data_x[:iteration, 0],
                      data_x[:iteration, 1], label="Data")
        ax[0].scatter(bayes_action[..., 0],
                      bayes_action[..., 1], label="Action")

        if "actions" in kwargs:
            actions = kwargs["actions"]
            actions = actions.cpu().detach().numpy()
            ax[0].scatter(
                actions[..., 0].reshape(-1, 1),
                actions[..., 1].reshape(-1, 1),
                label="Imaged Action",
            )
        if "X" in kwargs:
            for X in kwargs["X"][::-1]:
                ax[0].scatter(X[..., 0].reshape(-1, 1),
                              X[..., 1].reshape(-1, 1))

        ax[0].scatter(next_x[..., 0], next_x[..., 1], label="Next query")

        # Splotlight cost
        previous_x = buffer["x"][iteration - 1].squeeze()
        previous_x = previous_x.cpu().detach().numpy()
        splotlight = plt.Rectangle(
            (
                previous_x[0] - cfg.cost_func_hypers["radius"],
                previous_x[1] - cfg.cost_func_hypers["radius"],
            ),
            2 * cfg.cost_func_hypers["radius"],
            2 * cfg.cost_func_hypers["radius"],
            color="black",
            linestyle="dashed",
            fill=False,
        )
        ax[0].add_patch(splotlight)

    fig.legend()

    plt.savefig(f"{cfg.save_dir}/{iteration}.pdf")
    plt.close()

    return real_loss, bayes_action


def eval_and_plot_2D_without_posterior(
    func,
    wm,
    cfg,
    acqf,
    next_x,
    buffer,
    optimal_value,
    iteration,
    n_space=100,
    embedder=None,
    *args,
    **kwargs,
):
    r"""Evaluate and plot 2D function."""
    real_loss, bayes_action = eval_func(
        cfg, acqf, func, buffer, optimal_value, iteration
    )
    data_x = buffer["x"].cpu().detach()
    next_x = next_x.cpu().detach()

    # Plotting ###############################################################
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    if embedder is not None:
        bounds_plot_x = bounds_plot_y = [0, n_space - 1]
    else:
        bounds_plot_x, bounds_plot_y = cfg.bounds
    ax.set(xlabel="$x_1$", ylabel="$x_2$",
           xlim=bounds_plot_x, ylim=bounds_plot_y)
    # ax[1].set(xlabel="$x_1$", ylabel="$x_2$", xlim=bounds_plot, ylim=bounds_plot)
    if cfg.algo == "HES":
        title = "$\mathcal{H}_{\ell, \mathcal{A}}$-Entropy Search " + cfg.task
    elif cfg.algo == "qMSL":
        title = "Multi-Step Trees " + cfg.task
    else:
        title = cfg.algo + " " + cfg.task

    ax.set_title(label=title)
    # ax[1].set_title(label="Posterior mean")

    # Plot function in 2D ####################################################
    X_domain, Y_domain = cfg.bounds
    X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
    X, Y = np.meshgrid(X, Y)
    XY = torch.tensor(np.array([X, Y]))  # >> 2 x 100 x 100
    Z = func(XY.reshape(2, -1).T).reshape(X.shape)

    # Z_post = wm.posterior(XY.to(cfg.device, cfg.torch_dtype).permute(1, 2, 0)).mean
    # >> 100 x 100 x 1
    # Z_post = Z_post.squeeze(-1).cpu().detach()

    if embedder is not None:
        X, Y = (
            embedder.decode(XY.permute(1, 2, 0))
            .permute(2, 0, 1)
            .long()
            .cpu()
            .detach()
            .numpy()
        )

    cs = ax.contourf(X, Y, Z, levels=30, cmap="bwr", alpha=0.7)
    ax.set_aspect(aspect="equal")

    # cs1 = ax[1].contourf(X, Y, Z_post, levels=30, cmap="bwr", alpha=0.7)
    # ax[1].set_aspect(aspect="equal")

    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel("$f(x)$", rotation=270, labelpad=20)

    # cbar1 = ax[1].colorbar(cs1)
    # cbar1.set_ylabel("$\hat{f}(x)$", rotation=270, labelpad=20)

    # Plot data, optimal actions, next query #################################
    if embedder is not None:
        ax.scatter(*embedder.decode(data_x[:iteration]).long().T, label="Data")
        ax.scatter(*embedder.decode(bayes_action).long().T, label="Action")
        ax.scatter(*embedder.decode(next_x.unsqueeze(0)
                                    ).long().T, label="Next query")

        # Draw grid
        plt.vlines(np.arange(0, n_space), 0, n_space,
                   linestyles="dashed", alpha=0.05)
        plt.hlines(np.arange(0, n_space), 0, n_space,
                   linestyles="dashed", alpha=0.05)

        # Splotlight cost
        previous_x = embedder.decode(
            buffer["x"][iteration - 1].squeeze()).long()
        previous_x = previous_x.cpu().detach()
        splotlight = plt.Rectangle(
            (
                previous_x[0]
                - cfg.cost_func_hypers["radius"]
                * n_space
                / (cfg.bounds[0, 1] - cfg.bounds[0, 0]),
                previous_x[1]
                - cfg.cost_func_hypers["radius"]
                * n_space
                / (cfg.bounds[1, 1] - cfg.bounds[1, 0]),
            ),
            2
            * cfg.cost_func_hypers["radius"]
            * n_space
            / (cfg.bounds[0, 1] - cfg.bounds[0, 0]),
            2
            * cfg.cost_func_hypers["radius"]
            * n_space
            / (cfg.bounds[1, 1] - cfg.bounds[1, 0]),
            color="black",
            linestyle="dashed",
            fill=False,
        )
        ax.add_patch(splotlight)

        ax.set_xticks(range(0, n_space, 2))
        ax.set_yticks(range(0, n_space, 2))

    else:
        ax.scatter(data_x[:iteration, 0], data_x[:iteration, 1], label="Data")
        ax.scatter(bayes_action[..., 0], bayes_action[..., 1], label="Action")

        # actions = actions.cpu().detach().numpy()
        # ax[0].scatter(actions[..., 0].reshape(-1, 1), actions[..., 1].reshape(-1, 1), label="Imaged Action")
        ax.scatter(next_x[..., 0], next_x[..., 1], label="Next query")

        # Splotlight cost
        previous_x = buffer["x"][iteration - 1].squeeze()
        previous_x = previous_x.cpu().detach().numpy()
        splotlight = plt.Rectangle(
            (
                previous_x[0] - cfg.cost_func_hypers["radius"],
                previous_x[1] - cfg.cost_func_hypers["radius"],
            ),
            2 * cfg.cost_func_hypers["radius"],
            2 * cfg.cost_func_hypers["radius"],
            color="black",
            linestyle="dashed",
            fill=False,
        )
        ax.add_patch(splotlight)

    ax.legend()

    plt.savefig(f"{cfg.save_dir}/{iteration}.pdf")
    plt.close()

    return real_loss, bayes_action


def eval_and_plot_1D(
    func,
    wm,
    cfg,
    acqf,
    next_x,
    buffer,
    optimal_value,
    iteration,
    n_space=100,
    embedder=None,
    *args,
    **kwargs,
):
    r"""Evaluate and plot 1D function."""
    real_loss, bayes_action = eval_func(
        cfg, acqf, func, buffer, optimal_value, iteration
    )
    data_x = buffer["x"].cpu().detach().numpy()
    data_y = buffer["y"].cpu().detach().numpy()
    next_x = next_x.cpu().detach().numpy()

    # Plotting ###############################################################
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
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
    ub = np.max(upper)
    lb = np.min(lower)
    plt.vlines(data_x[iteration], lb, ub,
               color="black", label="current location")
    plt.vlines(
        data_x[iteration] - cfg.cost_func_hypers["radius"],
        lb,
        ub,
        color="black",
        linestyle="--",
    )
    plt.vlines(
        data_x[iteration] + cfg.cost_func_hypers["radius"],
        lb,
        ub,
        color="black",
        linestyle="--",
    )
    plt.vlines(next_x, lb, ub, color="red", label="optimal query")
    plt.vlines(bayes_action, lb, ub, color="blue", label="Bayes action")

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

    return real_loss, bayes_action


def eval_and_plot(func, wm, cfg, *args, **kwargs):
    r"""Draw the posterior of the model."""
    if cfg.x_dim == 1:
        return eval_and_plot_1D(func, wm, cfg, *args, **kwargs)
    elif cfg.x_dim == 2:
        return eval_and_plot_2D(func, wm, cfg, *args, **kwargs)
    else:
        print("Plotting is only done when x_dim is 1 or 2.")
        return eval_func


def draw_metric(
    save_dir, dict_metrics, env_names, algos, num_initial_points, num_steps
):
    r"""Draw the evaluation metric of all algorithms in all environments."""
    # Draw a figure with grid Nx3, each cell is a plot of all metrics
    # corresponding to all algorithms in one environment.
    # The number of rows is the number of environments divided by 3.
    assert len(env_names) != 0, "No environment to draw"

    num_rows = len(env_names) // 3
    if len(env_names) % 3 != 0:
        num_rows += 1
    fig, axs = plt.subplots(num_rows, 3, figsize=(12, 3 * num_rows))
    for i, env_name in enumerate(env_names):
        row = i // 3
        col = i % 3
        ax = axs[row * i + col]
        metrics = dict_metrics[env_name]
        for j, algo in enumerate(algos):
            nip = num_initial_points[env_name]
            ns = num_steps[env_name]
            mean = np.mean(metrics[j], axis=0)[nip - 1: ns]
            std = np.std(metrics[j], axis=0)[nip - 1: ns]
            # min_ = np.min(metrics[j], axis=0)[nip-1:ns]
            # max_ = np.max(metrics[j], axis=0)[nip-1:ns]
            x = list(range(nip - 1, ns))

            ax.plot(x, mean, label=algo)
            ax.fill_between(x, mean - std, mean + std, alpha=0.1)
            # ax.fill_between(x, min_, max_, alpha=0.1)

        ax.set_title(env_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Regret")

        if env_name == "SynGP":
            max_y = 20
        elif env_name == "HolderTable":
            max_y = 20
        elif env_name == "Alpine":
            max_y = 20
        ax.set_ylim([None, max_y])

        handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="outside upper center", ncol=8)
    plt.savefig(f"{save_dir}/eval_metric.pdf")
    plt.close()


def draw_loss_and_cost(save_dir, losses, costs, iteration):
    plt.plot(losses, label=f"Loss value")
    plt.plot(costs, label=f"Cost value")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Loss and cost value at iteration {iteration}")
    plt.legend()
    plt.savefig(f"{save_dir}/losses_and_costs_{iteration}.pdf")
    plt.close()
