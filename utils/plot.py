import matplotlib.pyplot as plt
import numpy as np
import torch
from tueplots import bundles

plt.rcParams.update(bundles.iclr2023())

color = {
    "C0": "#1f77b4",
    "C1": "#ff7f0e",
    "C2": "#2ca02c",
    "C3": "#d62728",
    "C4": "#9467bd",
    "C5": "#bcbd22",
    "C6": "#e377c2",
    "C7": "#17becf",
    # 'C8':'#bcbd22',
    # 'C9':'#17becf',
}

figsize = [7, 7]
plt.rcParams["figure.figsize"] = figsize


def eval_and_plot_2D(func, cfg, qhes, next_x, data, iteration):
    # Quality of the best decision from the current posterior distribution #############
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
        posterior = qhes.model.posterior(A_)
        fantasized_outcome = qhes.action_sampler(posterior)
        # >>> n_fantasy_at_action_pts x n_fantasy_at_design_pts
        # ... x batch_size x num_actions x 1

        fantasized_outcome = fantasized_outcome.squeeze(dim=-1)
        # >>> n_fantasy_at_action_pts x n_fantasy_at_design_pts
        # ... x batch_size x num_actions

        losses = -qhes.loss_function(A=A_, Y=fantasized_outcome)
        # >>> n_fantasy_at_design_pts x batch_size

        loss = losses.mean(dim=0).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 200 == 0:
            print(f"Eval optim round: {i}, Loss: {loss.item():.2f}")

    A = torch.sigmoid(A) * (ba_u - ba_l) + ba_l
    eval_metric = qhes.loss_function(A=A, Y=fantasized_outcome)
    eval_metric = eval_metric[0, 0].cpu().detach().numpy()
    optimal_action = A[0, 0].cpu().detach().numpy()
    value = qhes.loss_function(A=A, Y=func(A)[None, ...])[0, 0].cpu().detach().numpy()
    data_x = data.x.cpu().detach().numpy()
    next_x = next_x.cpu().detach().numpy()

    # Plotting #########################################################################
    fig, ax = plt.subplots(1, 1)
    bounds_plot = cfg.bounds
    ax.set(xlabel="$x_1$", ylabel="$x_2$", xlim=bounds_plot, ylim=bounds_plot)
    title = "$\mathcal{H}_{\ell, \mathcal{A}}$-Entropy Search " + cfg.task
    ax.set_title(label=title)

    # Plot function in 2D ##############################################################
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

    # Plot data, optimal actions, next query ###########################################
    ax.scatter(data_x[:, 0], data_x[:, 1], label="Data")
    ax.scatter(optimal_action[:, 0], optimal_action[:, 1], label="Action")
    ax.scatter(next_x[0], next_x[1], label="Next query")
    plt.legend()
    plt.savefig(f"{cfg.save_dir}/{iteration}.pdf")
    plt.close()

    return eval_metric, optimal_action, value


def eval_and_plot_1D(func, cfg, acqf, next_x, buffer, iteration, optimal_actions=None):
    if optimal_actions is not None:
        optimal_actions = optimal_actions.reshape(-1, optimal_actions.shape[-1])
        best_a = optimal_actions.numpy()
        plt.vlines(best_a, -5, 5, color="blue", label="optimal action", alpha=0.02)

    plt.vlines(buffer.x[-2], -5, 5, color="black", label="current location")
    plt.vlines(buffer.x[-2] - 0.1, -5, 5, color="black", linestyle="--")
    plt.vlines(buffer.x[-2] + 0.1, -5, 5, color="black", linestyle="--")
    plt.vlines(buffer.x[-1], -5, 5, color="red", label="optimal query")

    train_x = torch.linspace(-1, 1, 100).reshape(-1, 1).to(cfg.device)
    train_y = func(train_x)
    plt.plot(
        train_x.cpu().numpy(),
        train_y.cpu().numpy(),
        "black",
        alpha=0.2,
        label="Ground truth",
    )

    # compute posterior
    test_x = torch.linspace(-1, 1, 100).to(cfg.device)
    posterior = acqf.model.posterior(test_x)
    test_y = posterior.mean
    lower, upper = posterior.mvn.confidence_region()

    plt.plot(
        test_x.cpu().detach().numpy(),
        test_y.cpu().detach().numpy(),
        color["C1"],
        label="Posterior mean",
    )

    plt.fill_between(
        test_x.cpu().detach().numpy(),
        lower.cpu().detach().numpy(),
        upper.cpu().detach().numpy(),
        alpha=0.25,
        color=color["C2"],
    )

    # Scatter plot of all points using buffer.x and buffer.y with
    # gradient color from red to blue indicating the order of point in list
    plt.scatter(
        buffer.x,
        buffer.y,
        c=range(len(buffer.x)),
        cmap="Reds",
        marker="*",
        zorder=99,
    )
    # plt.tight_layout()
    plt.ylim(-5, 5)
    fig_name = f"{cfg.save_dir}/posterior_{iteration}.png"
    plt.savefig(fig_name, format="png")
    plt.close()


def draw_posterior(config, env, acqf, buffer, iteration, optimal_actions=None):
    print("Drawing posterior...")
    if config.x_dim == 1:
        eval_and_plot_1D(
            env,
            config,
            acqf,
            buffer.x[-1],
            buffer,
            iteration,
            optimal_actions=optimal_actions,
        )

    elif config.x_dim == 2:
        eval_and_plot_2D(env, config, acqf, buffer.x[-1], buffer, iteration)

    else:
        raise NotImplementedError


def draw_losses(config, losses, iteration):
    plt.plot(list(range(len(losses))), losses, label=f"Loss by iteration {iteration}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{config.save_dir}/losses_{iteration}.png", format="png")
    plt.close()


def draw_metric(save_dir, metrics, algos):
    if isinstance(metrics, list):
        metrics = np.array(metrics)

    plt.figure(figsize=(7, 7))
    for i, algo in enumerate(algos):
        mean = np.mean(metrics[i], axis=0)
        lower = np.min(metrics[i], axis=0)
        upper = np.max(metrics[i], axis=0)
        plt.plot(list(range(1, mean.shape[0] + 1)), mean, label=algo)

        plt.fill_between(list(range(1, mean.shape[0] + 1)), lower, upper, alpha=0.25)

    plt.xlabel("Iteration")
    plt.ylabel("Eval metric")
    fig_name = f"{save_dir}/eval_metric.png"
    plt.savefig(fig_name, format="png")
    plt.close()
