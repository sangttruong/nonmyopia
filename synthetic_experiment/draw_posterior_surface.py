import gpytorch

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from draw_metrics import get_env_info
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm
from tueplots import bundles
from utils import make_env, set_seed

plt.rcParams.update(bundles.iclr2024())

# ENVS = ["Ackley", "Alpine", "SynGP"]
ENVS = ["NightLight", "Ackley"]
KERNELS = ["RBF", "Linear", "Matern-0.5", "Matern-1.5", "Matern-2.5"]


def generate_initial_samples(bounds, x_dim, n_initial_points):
    bounds = np.array(bounds)
    if bounds.ndim < 2 or bounds.shape[0] < x_dim:
        bounds = np.tile(bounds, [x_dim, 1])

    local_bounds = np.zeros_like(bounds)
    local_bounds[..., 1] = 1

    n_partitions = int(n_initial_points ** (1 / x_dim))
    remaining_points = n_initial_points - n_partitions**x_dim
    ranges = np.linspace(local_bounds[..., 0], local_bounds[..., 1], n_partitions + 1).T
    range_bounds = np.stack((ranges[:, :-1], ranges[:, 1:]), axis=-1)
    cartesian_idxs = np.array(
        np.meshgrid(*([list(range(n_partitions))] * x_dim))
    ).T.reshape(-1, x_dim)
    cartesian_rb = range_bounds[list(range(x_dim)), cartesian_idxs]

    initial_points = np.concatenate(
        (
            np.random.uniform(
                low=cartesian_rb[..., 0],
                high=cartesian_rb[..., 1],
                size=[n_partitions**x_dim, x_dim],
            ),
            np.random.uniform(
                low=local_bounds[..., 0],
                high=local_bounds[..., 1],
                size=[remaining_points, x_dim],
            ),
        ),
        axis=0,
    )
    return initial_points


if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_space = 200

    fig, axs = plt.subplots(
        len(ENVS), len(KERNELS) + 1, figsize=(4 * (len(KERNELS) + 1), 4 * len(ENVS))
    )
    for ei, env_name in enumerate(ENVS):
        print(f"Drawing posterior surface for {env_name}")
        x_dim, bounds, radius, n_initial_points, algo_n_iterations = get_env_info(
            env_name, device
        )

        env = make_env(
            name=env_name,
            x_dim=x_dim,
            bounds=bounds,
            noise_std=0.0,
        )
        env = env.to(
            dtype=torch.float32,
            device=device,
        )

        bounds_plot_x, bounds_plot_y = np.array([0, 1]), np.array([0, 1])
        X_domain, Y_domain = bounds_plot_x, bounds_plot_y
        X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
        X, Y = np.meshgrid(X, Y)
        XY = torch.tensor(np.array([X, Y]))  # >> 2 x 100 x 100
        Z = env(XY.to(device, torch.float32).reshape(2, -1).T).reshape(X.shape)
        Z = Z.cpu().detach()

        axs[ei][0].set(
            xlabel="$x_1$", ylabel="$x_2$", xlim=bounds_plot_x, ylim=bounds_plot_y
        )
        axs[ei][0].contourf(X, Y, Z, levels=30, cmap="bwr", alpha=0.7)
        axs[ei][0].set_aspect(aspect="equal")
        axs[ei][0].set_ylabel(env_name, rotation=90, fontsize=30)

        if ei == 0:
            axs[ei][0].set_title("Ground Truth", fontsize=30)

        initial_points = generate_initial_samples(bounds.cpu(), x_dim, n_initial_points)
        initial_points = torch.tensor(initial_points, device=device).to(torch.float32)

        for ki, kernel_name in enumerate(tqdm(KERNELS)):
            if kernel_name == "RBF":
                kernel = None  # gpytorch.kernels.RBFKernel()
            elif kernel_name.startswith("Matern"):
                nu = float(kernel_name.split("-")[-1])
                kernel = gpytorch.kernels.MaternKernel(nu=nu)
            elif kernel_name == "Linear":
                kernel = gpytorch.kernels.LinearKernel()

            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=gpytorch.priors.NormalPrior(0, 1e-2)
            )

            surr_model = SingleTaskGP(
                train_X=initial_points,
                train_Y=env(initial_points).reshape(-1, 1),
                likelihood=likelihood,
                covar_module=kernel,
            )
            surr_model = surr_model.to(device=device)
            mll = ExactMarginalLogLikelihood(surr_model.likelihood, surr_model)
            fit_gpytorch_mll(mll)

            ax = axs[ei, ki + 1]
            ax.set(
                xlabel="$x_1$", ylabel="$x_2$", xlim=bounds_plot_x, ylim=bounds_plot_y
            )

            if ei == 0:
                ax.set_title(kernel_name, fontsize=30)

            Z_post = surr_model.posterior(
                XY.to(device, torch.float32).permute(1, 2, 0)
            ).mean
            Z_post = Z_post.squeeze(-1).cpu().detach()
            # >> 100 x 100 x 1

            ax.contourf(X, Y, Z_post, levels=30, cmap="bwr", alpha=0.7)
            ax.set_aspect(aspect="equal")

    plt.savefig(f"plots/posterior_surface.png", dpi=300)
    plt.close()
