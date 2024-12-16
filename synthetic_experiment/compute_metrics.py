import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
from argparse import ArgumentParser

import gpytorch

import numpy as np
import torch
import wandb
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from env_embedder import DiscreteEmbbeder
from gpytorch.mlls import ExactMarginalLogLikelihood
from main import make_env, Parameters
from tqdm import tqdm
from utils import eval_func, set_seed, str2bool


def compute_metrics(
    env,
    parms,
    buffer,
    embedder,
    device,
    surr_model_state_dicts,
):
    """Compute evaluation metrics for a given algorithm

    Args:
        env: Environment
        parms: Parameters
        buffer: Saved buffer
        embedder: Embedder for discretized environments. Otherwise, None
        device: Device. Either "cuda" or "cpu"
        surr_model_state_dicts: List of state_dicts for the surrogate model. If None, the surrogate model will be trained. Defaults to None.

    Returns:
        np.array: Shape (n_iterations, 3) where the columns are (u_observed, u_posterior, c_regret)
    """
    metrics = []
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=gpytorch.priors.NormalPrior(0, 1e-2)
    )
    final_run = len(surr_model_state_dicts) + parms.n_initial_points
    for iteration in tqdm(range(parms.n_initial_points - 1, final_run)):
        surr_model = SingleTaskGP(
            buffer["x"][: iteration + 1],
            buffer["y"][: iteration + 1],
            likelihood=likelihood,
            covar_module=parms.kernel,
        ).to(device, dtype=parms.torch_dtype)

        if iteration == final_run - 1:
            # Fit GP
            mll = ExactMarginalLogLikelihood(surr_model.likelihood, surr_model)
            fit_gpytorch_model(mll)
        else:
            surr_model.load_state_dict(
                surr_model_state_dicts[iteration - parms.n_initial_points + 1]
            )

        # Set surr_model to eval mode
        surr_model.eval()

        (u_observed, u_posterior, regret), A_chosen = eval_func(
            env, surr_model, parms, buffer, iteration, embedder
        )

        if iteration >= parms.n_initial_points:
            regret += metrics[-1][-1]  # Cummulative regret

        metrics.append([u_observed, u_posterior, regret])
        print(
            {"u_observed": u_observed, "u_posterior": u_posterior, "c_regret": regret}
        )
        wandb.log(
            {"u_observed": u_observed, "u_posterior": u_posterior, "c_regret": regret}
        )

    return np.array(metrics)


if __name__ == "__main__":
    wandb.init(project="nonmyopia-metrics")

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
    args = parser.parse_args()

    set_seed(args.seed)

    local_parms = Parameters(args)

    seed = args.seed
    env_name = args.env_name
    env_noise = args.env_noise
    env_discretized = args.env_discretized
    algo = args.algo
    cost_fn = args.cost_fn

    # Init environment
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

    # base_path = f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}"
    base_path = local_parms.save_dir

    print(f"Computing metrics for {algo}, {cost_fn}, seed{seed}")
    # os.path.exists(f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy"):
    if False:
        metrics = np.load(os.path.join(base_path, "/metrics.npy"))
    else:
        # Load buffer
        buffer_file = os.path.join(base_path, "buffer.pt")
        if not os.path.exists(buffer_file):
            raise RuntimeError(f"File {buffer_file} does not exist")
        buffer = torch.load(buffer_file, map_location=local_parms.device).to(
            dtype=local_parms.torch_dtype
        )

        # Load saved surr_model
        surr_model_state_dicts = []
        for step in range(local_parms.n_initial_points, local_parms.algo_n_iterations):
            surr_model_state_dict_file = os.path.join(
                base_path, f"surr_model_{step}.pt"
            )
            if not os.path.exists(surr_model_state_dict_file):
                break
                # raise RuntimeError(f"File {surr_model_state_dict_file} does not exist")

            surr_model_state_dicts.append(
                torch.load(surr_model_state_dict_file, map_location=local_parms.device)
            )

        if len(surr_model_state_dicts) != (
            local_parms.algo_n_iterations - local_parms.n_initial_points
        ):
            # raise RuntimeError(
            #     f"There are some error, please check saved models in {base_path}"
            # )
            current_run = len(surr_model_state_dicts) + local_parms.n_initial_points
            warnings.warn(
                f"The run has not been completed. Picking results from {current_run} / {local_parms.algo_n_iterations}."
            )

        # Initialize Embedder in case of discretization
        if local_parms.env_discretized:
            embedder = DiscreteEmbbeder(
                num_categories=local_parms.num_categories,
                bounds=local_parms.bounds,
            ).to(device=local_parms.device, dtype=local_parms.torch_dtype)
        else:
            embedder = None

        # Compute evaluation metrics
        metrics = compute_metrics(
            env=env,
            parms=local_parms,
            buffer=buffer,
            embedder=embedder,
            device=local_parms.device,
            surr_model_state_dicts=surr_model_state_dicts,
        )

        # Save results
        with open(
            os.path.join(local_parms.save_dir, "metrics.npy"),
            "wb",
        ) as f:
            np.save(f, metrics)

    wandb.finish()
