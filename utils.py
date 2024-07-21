import os
import torch
import random
import numpy as np
import dill as pickle
from tueplots import bundles
import matplotlib.pyplot as plt
from pathlib import Path
from botorch.test_functions.synthetic import (
    Ackley,  # XD Ackley function - Minimum
    Beale,  # 2D Beale function - Minimum
    Branin,  # 2D Branin function - Minimum
    Cosine8,  # 8D cosine function - Maximum,
    EggHolder,  # 2D EggHolder function - Minimum
    Griewank,  # XD Griewank function - Minimum
    Hartmann,  # 6D Hartmann function - Minimum
    HolderTable,  # 2D HolderTable function - Minimum
    Levy,  # XD Levy function - Minimum
    Powell,  # 4D Powell function - Minimum
    Rosenbrock,  # XD Rosenbrock function - Minimum
    SixHumpCamel,  # 2D SixHumpCamel function - Minimum
    StyblinskiTang,  # XD StyblinskiTang function - Minimum
)
from botorch.sampling.normal import SobolQMCNormalSampler

from acqfs import qBOAcqf
from synthetic_functions.alpine import AlpineN1
from synthetic_functions.syngp import SynGP
from env_wrapper import EnvWrapper

plt.rcParams.update(bundles.iclr2023())

def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(name, x_dim, bounds, noise_std=0.0):
    r"""Make environment."""
    if name == "Ackley":
        f_ = Ackley(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "Alpine":
        f_ = AlpineN1(dim=x_dim, noise_std=noise_std)
    elif name == "Beale":
        f_ = Beale(negate=True, noise_std=noise_std)
    elif name == "Branin":
        f_ = Branin(negate=True, noise_std=noise_std)
    elif name == "Cosine8":
        f_ = Cosine8(noise_std=noise_std)
    elif name == "EggHolder":
        f_ = EggHolder(negate=True, noise_std=noise_std)
    elif name == "Griewank":
        f_ = Griewank(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "Hartmann":
        f_ = Hartmann(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "HolderTable": 
        f_ = HolderTable(negate=True, noise_std=noise_std)
    elif name == "Levy":
        f_ = Levy(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "Powell":
        f_ = Powell(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "Rosenbrock":
        f_ = Rosenbrock(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "SixHumpCamel":
        f_ = SixHumpCamel(negate=True, noise_std=noise_std)
    elif name == "StyblinskiTang":
        f_ = StyblinskiTang(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "SynGP":
        f_ = SynGP(dim=x_dim, noise_std=noise_std)
    else:
        raise NotImplementedError

    # Set env bound
    f_.bounds[0, :] = bounds[..., 0]
    f_.bounds[1, :] = bounds[..., 1]

    # Wrapper for normalizing output
    f = EnvWrapper(name, f_)
    f.noise_std = noise_std * (f.range_y[1] - f.range_y[0]) /  100
    
    return f


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_save_dir(config):
    r"""Create save directory without overwriting directories."""
    init_dir_path = Path(config.save_dir)
    dir_path = Path(str(init_dir_path))

    if not os.path.exists(os.path.join(config.save_dir, "buffer.pt")):
        config.cont = False
    
    if not config.cont and not os.path.exists(config.save_dir):
        dir_path.mkdir(parents=True, exist_ok=False)
    elif not config.cont and os.path.exists(config.save_dir):
        config.save_dir = str(dir_path)
    elif config.cont and not os.path.exists(config.save_dir):
        print(f"WARNING: save_dir={config.save_dir} does not exist. Rerun from scratch.")
        dir_path.mkdir(parents=True, exist_ok=False)
        
    print(f"Created save_dir: {config.save_dir}")

    # Save config to save_dir as parameters.json
    config_path = dir_path / "parameters.json"
    with open(str(config_path), "w", encoding="utf-8") as file_handle:
        config_dict = str(config)
        file_handle.write(config_dict)


def eval_and_plot(env, model, parms, *args, **kwargs):
    r"""Draw the posterior of the model."""
    if parms.plot:
        if parms.x_dim == 1:
            return eval_and_plot_1D(env, model, parms, *args, **kwargs)
        elif parms.x_dim == 2:
            return eval_and_plot_2D(env, model, parms, *args, **kwargs)
        else:
            print("NOTICE: Plotting is only done when x_dim is 1 or 2.")
            return eval_func(env, model, parms, *args, **kwargs)
    else:
        return eval_func(env, model, parms, *args, **kwargs)


def eval_func(
    env, model, parms, buffer, iteration, embedder=None, *args, **kwargs
):
    # Quality of the best decision from the current posterior distribution ###
    cost_fn = parms.cost_function_class(**parms.cost_func_hypers)
    previous_cost = buffer["cost"][:iteration+1].sum() if iteration+1 > parms.n_initial_points else 0.0
        
    u_observed = torch.max(buffer["y"][:iteration+1]).item()
    
    ######################################################################
    
    if parms.algo.startswith("HES"):
        # Initialize A consistently across fantasies
        A = buffer["x"][iteration].clone().repeat(
            parms.n_restarts, parms.n_actions, 1)
        A = A + torch.randn_like(A) * 0.01
        if embedder is not None:
            A = embedder.decode(A)
            A = torch.nn.functional.one_hot(A, num_classes=parms.num_categories).to(parms.torch_dtype)
        A.requires_grad = True

        # Initialize optimizer
        optimizer = torch.optim.AdamW([A], lr=parms.acq_opt_lr)
        loss_fn = parms.loss_function_class(**parms.loss_func_hypers)

        for i in range(1000):
            if embedder is not None:
                actions = embedder.encode(A)
            else:
                actions = A
            ppd = model(actions)
            y_A = ppd.rsample()

            losses = loss_fn(A=actions, Y=y_A) + cost_fn(
                prev_X=buffer["x"][iteration].expand_as(actions), current_X=actions, previous_cost=previous_cost
            )
            # >>> n_fantasy x batch_size

            loss = losses.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (i+1) % 200 == 0:
                print(f"Eval optim round: {i+1}, Loss: {loss.item():.2f}")

        aidx = losses.squeeze(-1).argmin()
        
    else:
        # Construct acqf
        sampler = SobolQMCNormalSampler(
            sample_shape=1, seed=0, resample=False
        )
        nf_design_pts = [1]
        
        acqf = qBOAcqf(
            name=parms.algo,
            model=model, 
            lookahead_steps=0 if parms.algo_lookahead_steps == 0 else 1,
            n_actions=parms.n_actions,
            n_fantasy_at_design_pts=nf_design_pts,
            loss_function_class=parms.loss_function_class,
            loss_func_hypers=parms.loss_func_hypers,
            cost_function_class=parms.cost_function_class,
            cost_func_hypers=parms.cost_func_hypers,
            sampler=sampler,
            best_f=buffer["y"][:iteration+1].max(),
        )

        maps = []
        if parms.algo_lookahead_steps > 0:
            x = buffer["x"][iteration].clone().repeat(
                parms.n_restarts, 1)
            if embedder is not None:
                x = embedder.decode(x)
                x = torch.nn.functional.one_hot(x, num_classes=parms.num_categories).to(parms.torch_dtype)
            maps.append(x)

        A = buffer["x"][iteration].clone().repeat(
            parms.n_restarts * parms.n_actions, 1)
        A = A + torch.randn_like(A) * 0.01
        if embedder is not None:
            A = embedder.decode(A)
            A = torch.nn.functional.one_hot(A, num_classes=parms.num_categories).to(parms.torch_dtype)
        A.requires_grad = True
        maps.append(A)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW([A], lr=parms.acq_opt_lr)

        # Get prevX, prevY
        prev_X = buffer["x"][iteration: iteration+1].expand(
            parms.n_restarts, -1
        )
        if embedder is not None:
            # Discretize: Continuous -> Discrete
            prev_X = embedder.decode(prev_X)
            prev_X = torch.nn.functional.one_hot(
                prev_X, num_classes=parms.num_categories
            ).to(dtype=parms.torch_dtype)
            # >>> n_restarts x x_dim x n_categories

        prev_y = buffer["y"][iteration: iteration+1].expand(
            parms.n_restarts, -1
        ).to(dtype=parms.torch_dtype)

        for i in range(1000):
            return_dict = acqf.forward(
                prev_X=prev_X,
                prev_y=prev_y,
                prev_hid_state=None,
                maps=maps,
                embedder=embedder,
                prev_cost=previous_cost
            )
            
            losses = (return_dict["acqf_loss"] + return_dict["acqf_cost"])
            # >>> n_fantasy_at_design_pts x batch_size

            loss = losses.mean()
            grads = torch.autograd.grad(
                loss, [A], allow_unused=True)
            for param, grad in zip([A], grads):
                param.grad = grad
            optimizer.step()
            optimizer.zero_grad()

            if (i+1) % 200 == 0:
                print(f"Eval optim round: {i}, Loss: {loss.item():.2f}")

        aidx = losses.squeeze(-1).argmin()
        if embedder is not None:
            A = A.reshape(parms.n_restarts, parms.n_actions, parms.x_dim, parms.num_categories)
        else:
            A = A.reshape(parms.n_restarts, parms.n_actions, parms.x_dim)
    
    if embedder is not None:
        A_chosen = embedder.encode(A[aidx]).detach()
    else:
        A_chosen = A[aidx].detach()
    u_posterior = env(A_chosen).item()

    ######################################################################
    regret = 1 - u_posterior

    return (u_observed, u_posterior, regret), A_chosen.cpu()


def eval_and_plot_2D(
    env,
    model,
    parms,
    next_x,
    buffer,
    iteration,
    n_space=100,
    embedder=None,
    *args,
    **kwargs,
):
    r"""Evaluate and plot 2D function."""
    real_loss, bayes_action = eval_func(
        env, model, parms, buffer, iteration, embedder=embedder
    )

    data_x = buffer["x"].cpu().detach()
    next_x = next_x.cpu().detach()

    # Plotting ###############################################################
    if iteration < parms.n_initial_points:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        surface_ax = surface_ax
        posterior_ax = ax[1]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        surface_ax = ax
        posterior_ax = None
        
    if embedder is not None:
        bounds_plot_x = bounds_plot_y = [0, n_space - 1]
    else:
        bounds_plot_x, bounds_plot_y = parms.bounds.cpu().numpy()
    surface_ax.set(xlabel="$x_1$", ylabel="$x_2$",
              xlim=bounds_plot_x, ylim=bounds_plot_y)

    if parms.algo == "HES":
        title = "$\mathcal{H}_{\ell, \mathcal{A}}$-Entropy Search " + parms.task
    elif parms.algo == "qMSL":
        title = "Multi-Step Trees " + parms.task
    else:
        title = parms.algo + " " + parms.task

    surface_ax.set_title(label=title)

    # Plot function in 2D ####################################################
    X_domain, Y_domain = parms.bounds.cpu().numpy()
    X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
    X, Y = np.meshgrid(X, Y)
    XY = torch.tensor(np.array([X, Y]))  # >> 2 x 100 x 100
    Z = env(XY.reshape(2, -1).T).reshape(X.shape)

    Z_post = model.posterior(
        XY.to(parms.device, parms.torch_dtype).permute(1, 2, 0)).mean
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

    cs = surface_ax.contourf(X, Y, Z, levels=30, cmap="bwr", alpha=0.7)
    surface_ax.set_aspect(aspect="equal")
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel("$f(x)$", rotation=270, labelpad=20)

    if posterior_ax is not None:
        posterior_ax.set(xlabel="$x_1$", ylabel="$x_2$", xlim=bounds_plot_x, ylim=bounds_plot_y)
        posterior_ax.set_title(label="Posterior mean")
        cs1 = posterior_ax.contourf(X, Y, Z_post, levels=30, cmap="bwr", alpha=0.7)
        posterior_ax.set_aspect(aspect="equal")
        cbar1 = fig.colorbar(cs1)
        cbar1.ax.set_ylabel("$\hat{f}(x)$", rotation=270, labelpad=20)

    # Plot data, optimal actions, next query #################################
    if embedder is not None:
        surface_ax.scatter(
            *embedder.decode(data_x[:iteration].to(device=embedder.device)).cpu().long().T, label="Data")
        surface_ax.scatter(*embedder.decode(bayes_action.to(device=embedder.device)).cpu().long().T, label="Action")
        surface_ax.scatter(
            *embedder.decode(next_x.unsqueeze(0).to(device=embedder.device)).cpu().long().T, label="Next query"
        )

        # Draw grid
        plt.vlines(np.arange(0, n_space), 0, n_space,
                   linestyles="dashed", alpha=0.05)
        plt.hlines(np.arange(0, n_space), 0, n_space,
                   linestyles="dashed", alpha=0.05)

        # Splotlight cost
        previous_x = embedder.decode(
            buffer["x"][iteration - 1].squeeze().to(device=embedder.device)).long()
        previous_x = previous_x.cpu().detach()
        splotlight = plt.Rectangle(
            (
                (previous_x[0]
                - parms.cost_func_hypers["radius"]
                * n_space
                / (parms.bounds[0, 1] - parms.bounds[0, 0])).cpu(),
                (previous_x[1]
                - parms.cost_func_hypers["radius"]
                * n_space
                / (parms.bounds[1, 1] - parms.bounds[1, 0])).cpu(),
            ),
            (2
            * parms.cost_func_hypers["radius"]
            * n_space
            / (parms.bounds[0, 1] - parms.bounds[0, 0])).cpu(),
            (2
            * parms.cost_func_hypers["radius"]
            * n_space
            / (parms.bounds[1, 1] - parms.bounds[1, 0])).cpu(),
            color="black",
            linestyle="dashed",
            fill=False,
        )
        surface_ax.add_patch(splotlight)

        surface_ax.set_xticks(range(0, n_space, 2))
        surface_ax.set_yticks(range(0, n_space, 2))

    else:
        surface_ax.scatter(data_x[:iteration, 0],
                      data_x[:iteration, 1], label="Data")
        surface_ax.scatter(bayes_action[..., 0],
                      bayes_action[..., 1], label="Action")

        if "actions" in kwargs and kwargs["actions"] is not None:
            actions = kwargs["actions"]
            actions = actions.cpu().detach().numpy()
            surface_ax.scatter(
                actions[..., 0].reshape(-1, 1),
                actions[..., 1].reshape(-1, 1),
                label="Imaged Action",
            )
        if "X" in kwargs:
            for X in kwargs["X"].cpu().numpy()[::-1]:
                surface_ax.scatter(X[..., 0].reshape(-1, 1),
                              X[..., 1].reshape(-1, 1))

        surface_ax.scatter(next_x[..., 0], next_x[..., 1], label="Next query")

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
        surface_ax.add_patch(splotlight)

    fig.legend()

    plt.savefig(f"{parms.save_dir}/{iteration}.pdf")
    plt.close()

    return real_loss, bayes_action


def eval_and_plot_1D(
    env,
    model,
    parms,
    next_x,
    buffer,
    iteration,
    n_space=100,
    embedder=None,
    *args,
    **kwargs,
):
    r"""Evaluate and plot 1D function."""
    real_loss, bayes_action = eval_func(
        env, model, parms, buffer, iteration, embedder=embedder
    )
    data_x = buffer["x"].cpu().detach().numpy()
    data_y = buffer["y"].cpu().detach().numpy()
    next_x = next_x.cpu().detach().numpy()

    # Plotting ###############################################################
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    bounds_plot = parms.bounds.cpu().numpy()
    y_bounds = (-5, 5)
    ax.set(xlabel="$x$", ylabel="$y$", xlim=bounds_plot, ylim=y_bounds)
    title = "$\mathcal{H}_{\ell, \mathcal{A}}$-Entropy Search " + parms.task
    ax.set_title(label=title)

    # Plot function and posterior in 1D #######################################
    x = torch.linspace(-1, 1, 100).reshape(-1, 1)
    posterior = model.posterior(x)
    test_y = posterior.mean
    lower, upper = posterior.mvn.confidence_region()
    x = x.cpu().detach().numpy()
    test_y = test_y.cpu().detach().numpy()
    lower = lower.cpu().detach().numpy()
    upper = upper.cpu().detach().numpy()
    plt.plot(x, env(x).numpy(), alpha=0.2, label="Ground truth")
    plt.plot(x, test_y, label="Posterior mean")
    plt.fill_between(x, lower, upper, alpha=0.25)

    # Plot data, optimal actions, next query #################################
    ub = np.max(upper)
    lb = np.min(lower)
    plt.vlines(data_x[iteration], lb, ub,
               color="black", label="current location")
    plt.vlines(
        data_x[iteration] - parms.cost_func_hypers["radius"],
        lb,
        ub,
        color="black",
        linestyle="--",
    )
    plt.vlines(
        data_x[iteration] + parms.cost_func_hypers["radius"],
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
    plt.savefig(f"{parms.save_dir}/posterior_{iteration}.pdf")
    plt.close()

    return real_loss, bayes_action


def draw_loss_and_cost(save_dir, losses, costs, iteration):
    plt.plot(losses, label=f"Loss value")
    plt.plot(costs, label=f"Cost value")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Loss and cost value at iteration {iteration}")
    plt.legend()
    plt.savefig(f"{save_dir}/losses_and_costs_{iteration}.pdf")
    plt.close()

