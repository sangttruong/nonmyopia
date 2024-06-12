# This file is made for analyzing world models
from utils import set_seed, make_env
import matplotlib.pyplot as plt
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_model
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import copy
import os
import pandas as pd
from tueplots import bundles
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.kernels import MaternKernel, ScaleKernel

plt.rcParams.update(bundles.neurips2023())


class Parameters:
    r"""Class to store all parameters for the experiment."""

    def __init__(self, args):
        r"""Initialize parameters."""
        # general parameters
        if torch.cuda.is_available():
            self.device = f"cuda:{args.gpu_id}"
        else:
            self.device = "cpu"

        print("Using device:", self.device)

        self.gpu_id = args.gpu_id
        self.torch_dtype = torch.float32
        self.seed = args.seed

        self.env_name = args.env_name
        self.env_noise = args.env_noise
        self.y_dim = 1
        self.save_dir = f"gp_results/{args.env_name}_noise{args.env_noise}"

        if args.env_discretized:
            self.env_discretized = True
            self.num_categories = 20
            self.save_dir += "_env_discretized"
        else:
            self.env_discretized = False
            self.num_categories = None

        self.kernel = None

        if self.env_name == "Ackley":
            self.x_dim = 2
            self.bounds = [-2, 2]
            self.radius = 0.3

        elif self.env_name == "Alpine":
            self.x_dim = 2
            self.bounds = [0, 10]
            self.radius = 0.75

        elif self.env_name == "Beale":
            self.x_dim = 2
            self.bounds = [-4.5, 4.5]
            self.radius = 0.65

        elif self.env_name == "Branin":
            self.x_dim = 2
            self.bounds = [[-5, 10], [0, 15]]
            self.radius = 1.2

        elif self.env_name == "Cosine8":
            self.x_dim = 8
            self.bounds = [-1, 1]
            self.radius = 0.3

        elif self.env_name == "EggHolder":
            self.x_dim = 2
            self.bounds = [-100, 100]
            self.radius = 15.0

        elif self.env_name == "Griewank":
            self.x_dim = 2
            self.bounds = [-600, 600]
            self.radius = 85.0

        elif self.env_name == "Hartmann":
            self.x_dim = 6
            self.bounds = [0, 1]
            self.radius = 0.15

        elif self.env_name == "HolderTable":
            self.x_dim = 2
            self.bounds = [0, 10]
            self.radius = 0.75

        elif self.env_name == "Levy":
            self.x_dim = 2
            self.bounds = [-10, 10]
            self.radius = 1.5

        elif self.env_name == "Powell":
            self.x_dim = 4
            self.bounds = [-4, 5]
            self.radius = 0.9

        elif self.env_name == "SixHumpCamel":
            self.x_dim = 2
            self.bounds = [[-3, 3], [-2, 2]]
            self.radius = 0.4

        elif self.env_name == "StyblinskiTang":
            self.x_dim = 2
            self.bounds = [-5, 5]
            self.radius = 0.75

        elif self.env_name == "SynGP":
            self.x_dim = 2
            self.bounds = [-1, 1]
            self.radius = 0.15

        else:
            raise NotImplementedError

        # Random select initial points
        self.bounds = np.array(self.bounds)
        if self.bounds.ndim < 2 or self.bounds.shape[0] < self.x_dim:
            self.bounds = np.tile(self.bounds, [self.x_dim, 1])
        self.bounds = torch.tensor(self.bounds).to(self.device)

    def __str__(self):
        r"""Return string representation of parameters."""
        output = []
        for k in self.__dict__.keys():
            output.append(f"{k}: {self.__dict__[k]}")
        return "\n".join(output)


def eval(
    func,
    wm,
    cfg,
    eval_fns,
    data_x=None,
    n_points=10000
):
    r"""Evaluate function."""

    # Evaluate ###############################################################
    if data_x is None:
        data_x = np.random.uniform(
            low=cfg.bounds[..., 0].cpu(),
            high=cfg.bounds[..., 1].cpu(),
            size=[n_points, local_parms.x_dim]
        )
        data_x = torch.tensor(data_x, dtype=torch.float32, device=cfg.device)

    data_y = func(data_x).reshape(-1)
    y_hat = wm.posterior(data_x).mean.detach()
    y_hat = y_hat.reshape(-1)

    # Evaluate functions #####################################################
    eval_vals = []
    for eval_fn in eval_fns:
        val = eval_fn(data_y, y_hat).item()
        eval_vals.append(val)

    return eval_vals


def plot(
    func,
    wm,
    cfg,
    data_x,
    n_space=100,
    embedder=None,
):
    r"""Evaluate and plot 2D function."""

    # Plotting ###############################################################
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    if embedder is not None:
        bounds_plot_x = bounds_plot_y = [0, n_space - 1]
    else:
        bounds_plot_x, bounds_plot_y = cfg.bounds.cpu()

    ax[0].set(xlabel="$x_1$", ylabel="$x_2$",
              xlim=bounds_plot_x, ylim=bounds_plot_y)
    ax[1].set(xlabel="$x_1$", ylabel="$x_2$",
              xlim=bounds_plot_x, ylim=bounds_plot_y)

    ax[0].set_title(label=cfg.env_name)
    ax[1].set_title(label="Posterior mean")

    # Plot function in 2D ####################################################
    X_domain, Y_domain = cfg.bounds.cpu()
    X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
    X, Y = np.meshgrid(X, Y)
    XY = torch.tensor(np.array([X, Y]))  # >> 2 x 100 x 100
    Z = func(XY.reshape(2, -1).T).reshape(X.shape)

    Z_post = wm.posterior(
        XY.to(cfg.device, cfg.torch_dtype).permute(1, 2, 0)).mean
    # >> 100 x 100 x 1
    Z_post = Z_post.squeeze(-1).cpu().detach()

    # Plot data
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

    if embedder is not None:
        ax[0].scatter(*embedder.decode(data_x).long().T, label="Data")

        # Draw grid
        plt.vlines(np.arange(0, n_space), 0, n_space,
                   linestyles="dashed", alpha=0.05)
        plt.hlines(np.arange(0, n_space), 0, n_space,
                   linestyles="dashed", alpha=0.05)

        ax[0].set_xticks(range(0, n_space, 2))
        ax[0].set_yticks(range(0, n_space, 2))

    else:
        ax[0].scatter(data_x[..., 0],
                      data_x[..., 1], label="Data")

    fig.legend()

    plt.savefig(
        f"{cfg.save_dir}/{cfg.env_name}_npoints{cfg.n_points}_seed{cfg.seed}.pdf")
    plt.close()


def main(env, local_parms):
    # Random select initial points
    # bounds = np.array(local_parms.bounds.cpu())
    # if bounds.shape[0] < local_parms.x_dim:
    #     bounds = np.tile(bounds, [local_parms.x_dim, 1])
    bounds = local_parms.bounds.cpu().numpy()

    n_partitions = int(local_parms.n_points ** (1 / local_parms.x_dim))
    remaining_points = local_parms.n_points - n_partitions**local_parms.x_dim

    ranges = np.linspace(bounds[..., 0], bounds[..., 1], n_partitions+1).T
    range_bounds = np.stack((ranges[:, :-1], ranges[:, 1:]), axis=-1)
    cartesian_idxs = np.array(np.meshgrid(*([list(range(n_partitions))] * local_parms.x_dim))).T.reshape(
        -1, local_parms.x_dim
    )
    cartesian_rb = range_bounds[list(range(local_parms.x_dim)), cartesian_idxs]
    train_X = np.concatenate(
        (
            np.random.uniform(
                low=cartesian_rb[..., 0],
                high=cartesian_rb[..., 1],
                size=[n_partitions**local_parms.x_dim, local_parms.x_dim],
            ),
            np.random.uniform(
                low=bounds[..., 0],
                high=bounds[..., 1],
                size=[remaining_points, local_parms.x_dim],
            ),
        ),
        axis=0,
    )

    train_X = torch.tensor(
        train_X, dtype=local_parms.torch_dtype, device=local_parms.device)
    train_y = env(train_X).reshape(-1, 1)
    
    WM = SingleTaskGP(
        train_X=train_X,
        train_Y=train_y,
    )
    
    mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
    fit_gpytorch_model(mll)

    def rmse_fn(y, yhat):
        return torch.sqrt(
            torch.nn.functional.mse_loss(y, yhat))

    def rsquare_fn(y, yhat):
        # Calculate residual sum of squares (RSS)
        RSS_val = torch.sum((y - yhat) ** 2)

        # Calculate total sum of squares (TSS)
        TSS_val = torch.sum((y - y.mean()) ** 2)

        # Calculate R^2
        R2_val = 1 - (RSS_val / TSS_val)
        return R2_val

    train_eval_vals = eval(
        func=env,
        wm=WM,
        cfg=local_parms,
        eval_fns=[rmse_fn, rsquare_fn],
        data_x=train_X,
    )

    test_eval_vals = eval(
        func=env,
        wm=WM,
        cfg=local_parms,
        eval_fns=[rmse_fn, rsquare_fn],
    )

    if local_parms.x_dim == 2:
        plot(
            func=env,
            wm=WM,
            cfg=local_parms,
            data_x=train_X.cpu(),
        )

    return train_eval_vals, test_eval_vals


def plot_means_stds(
    data,
    save_file
):
    fig, ax = plt.subplots()
    for env_name in data.keys():
        train_means = data[env_name]["train_means"]
        train_stds = data[env_name]["train_stds"]
        test_means = data[env_name]["test_means"]
        test_stds = data[env_name]["test_stds"]

        ax.plot(
            list(train_means.keys()),
            np.array(list(train_means.values()))[..., 1],
            label=f"{env_name} - training",
        )
        ax.fill_between(
            list(train_means.keys()),
            np.array(list(train_means.values()))[..., 1] -
            np.array(list(train_stds.values()))[..., 1],
            np.array(list(train_means.values()))[..., 1] +
            np.array(list(train_stds.values()))[..., 1],
            alpha=0.2,
        )

        ax.plot(
            list(test_means.keys()),
            np.array(list(test_means.values()))[..., 1],
            label=f"{env_name} - testing",
            linestyle="--",
        )
        ax.fill_between(
            list(test_means.keys()),
            np.array(list(test_means.values()))[..., 1] -
            np.array(list(test_stds.values()))[..., 1],
            np.array(list(test_means.values()))[..., 1] +
            np.array(list(test_stds.values()))[..., 1],
            alpha=0.1,
        )

    ax.set_xlabel("Number of samples")
    ax.set_ylabel("$R^2$")
    ax.legend()
    plt.savefig(save_file)
    plt.close()


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[2, 3, 5, 7, 11])
    parser.add_argument("--env_names", nargs="+", type=str, default=["SynGP"])
    parser.add_argument("--env_noise", type=float, default=0.0)
    parser.add_argument("--env_discretized", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    eval_metrics = {}
    for e, env_name in enumerate(args.env_names):
        print("Env name: ", env_name)
        args.seed = None
        args.env_name = env_name
        local_parms = Parameters(args)

        env = make_env(
            name=local_parms.env_name,
            x_dim=local_parms.x_dim,
            bounds=local_parms.bounds,
            noise_std=local_parms.env_noise,
        )
        env = env.to(
            dtype=local_parms.torch_dtype,
            device=local_parms.device,
        )

        train_means = {}
        train_stds = {}
        test_means = {}
        test_stds = {}

        if os.path.exists(f"gp_results/{env_name}_noise{args.env_noise}/{env_name}_train.csv") and os.path.exists(f"gp_results/{env_name}_noise{args.env_noise}/{env_name}_test.csv"):
            train_df = pd.read_csv(
                f"gp_results/{env_name}_noise{args.env_noise}/{env_name}_train.csv")
            train_means = {p: [x, y] for p, x, y in zip(
                train_df["n_points"].values, train_df["rmse_mean"].values, train_df["rsquare_mean"].values)}
            train_stds = {p: [x, y] for p, x, y in zip(
                train_df["n_points"].values, train_df["rmse_std"].values, train_df["rsquare_std"].values)}

            test_df = pd.read_csv(
                f"gp_results/{env_name}_noise{args.env_noise}/{env_name}_test.csv")
            test_means = {p: [x, y] for p, x, y in zip(
                test_df["n_points"].values, test_df["rmse_mean"].values, test_df["rsquare_mean"].values)}
            test_stds = {p: [x, y] for p, x, y in zip(
                test_df["n_points"].values, test_df["rmse_std"].values, test_df["rsquare_std"].values)}
        else:
            for n_points in tqdm([1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]):
            # for n_points in tqdm([100, 500]):
                list_train_vals = []
                list_test_vals = []
                for i, seed in enumerate(args.seeds):
                    set_seed(seed)
                    local_parms.seed = seed
                    local_parms.n_points = n_points

                    init_dir_path = Path(local_parms.save_dir)
                    dir_path = Path(str(init_dir_path))
                    dir_path.mkdir(parents=True, exist_ok=True)

                    train_vals, text_vals = main(env, local_parms)
                    list_train_vals.append(train_vals)
                    list_test_vals.append(text_vals)

                list_train_vals = np.array(list_train_vals)
                list_test_vals = np.array(list_test_vals)
                train_means[n_points] = np.mean(list_train_vals, axis=0)
                train_stds[n_points] = np.std(list_train_vals, axis=0)
                test_means[n_points] = np.mean(list_test_vals, axis=0)
                test_stds[n_points] = np.std(list_test_vals, axis=0)

            # Save means and stds
            save_path = f"{local_parms.save_dir}/{local_parms.env_name}_train.csv"
            with open(save_path, "w") as f:
                f.write("n_points,rmse_mean,rmse_std,rsquare_mean,rsquare_std\n")
                for n_points in train_means.keys():
                    f.write(
                        f"{n_points},{train_means[n_points][0]},{train_stds[n_points][0]},{train_means[n_points][1]},{train_stds[n_points][1]}\n")

            save_path = f"{local_parms.save_dir}/{local_parms.env_name}_test.csv"
            with open(save_path, "w") as f:
                f.write("n_points,rmse_mean,rmse_std,rsquare_mean,rsquare_std\n")
                for n_points in test_means.keys():
                    f.write(
                        f"{n_points},{test_means[n_points][0]},{test_stds[n_points][0]},{test_means[n_points][1]},{test_stds[n_points][1]}\n")

            print(f"Saved to {save_path}")

        eval_metrics[env_name] = {
            "train_means": train_means,
            "train_stds": train_stds,
            "test_means": test_means,
            "test_stds": test_stds,
        }

        # Plot
        plot_path = f"gp_results/{env_name}_noise{args.env_noise}/{env_name}_r2.pdf"
        plot_means_stds({env_name: eval_metrics[env_name]}, plot_path)
        print(f"Saved env plot to {plot_path}")

    # Plot
    if len(args.env_names) > 1:
        plot_path = f"gp_results/{'+'.join(args.env_names)}_noise{args.env_noise}_r2.pdf"
        plot_means_stds(eval_metrics, plot_path)
        print(f"Saved plot to {plot_path}")
