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
from main import Parameters, make_env
from acqfs import qBOAcqf
from utils import set_seed, str2bool, eval_func
# from synthetic_functions.alpine import AlpineN1
# from synthetic_functions.syngp import SynGP
from env_embedder import DiscreteEmbbeder
from env_wrapper import EnvWrapper

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from tqdm import tqdm
import os
import argparse
from tensordict import TensorDict
from pathlib import Path
from argparse import ArgumentParser
import wandb

# Compute metrics


def compute_metrics(
    env,
    parms,
    buffer,
    embedder,
    device,
    surr_model_state_dicts=None,
):
    metrics = []

    for iteration in tqdm(range(parms.n_initial_points-1, parms.algo_n_iterations)):
        surr_model = SingleTaskGP(
            buffer["x"][:iteration+1],
            buffer["y"][:iteration+1],
            # input_transform=Normalize(
            #     d=parms.x_dim, bounds=parms.bounds.T),
            # outcome_transform=Standardize(1),
        ).to(device, dtype=parms.torch_dtype)

        if iteration == parms.algo_n_iterations - 1:
            # Fit GP
            mll = ExactMarginalLogLikelihood(surr_model.likelihood, surr_model)
            fit_gpytorch_model(mll)
        else:
            surr_model.load_state_dict(
                surr_model_state_dicts[iteration - parms.n_initial_points + 1])

        # Set surr_model to eval mode
        surr_model.eval()

        (u_observed, u_posterior, regret), A_chosen = eval_func(
            env, surr_model, parms, buffer, iteration, embedder)

        if iteration >= parms.n_initial_points:
            regret += metrics[-1][-1]  # Cummulative regret

        metrics.append([u_observed, u_posterior, regret])
        print({"u_observed": u_observed,
              "u_posterior": u_posterior, "c_regret": regret})
        # wandb.log({"u_observed": u_observed, "u_posterior": u_posterior, "c_regret": regret})

    return np.array(metrics)


if __name__ == '__main__':
    # wandb.init(project="nonmyopia-metrics")

    # Parse args
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--task", type=str, default="topk")
    parser.add_argument("--env_name", type=str, default="SynGP")
    parser.add_argument("--env_noise", type=float, default=0.0)
    parser.add_argument("--env_discretized", type=str2bool, default=False)
    parser.add_argument("--algo", type=str, default="HES-TS-AM-20")
    parser.add_argument("--cost_fn", type=str, default="euclidean")
    parser.add_argument("--plot", type=str2bool, default=False)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--cont", type=str2bool, default=True)
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

    print(f"Computing metrics for {algo}, {cost_fn}, seed{seed}")
    # os.path.exists(f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy"):
    if False:
        metrics = np.load(
            f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy")
    else:
        buffer_file = f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/buffer.pt"
        if not os.path.exists(buffer_file):
            raise RuntimeError(f"File {buffer_file} does not exist")
        buffer = torch.load(buffer_file, map_location=local_parms.device).to(
            dtype=local_parms.torch_dtype)

        surr_model_state_dicts = []
        for step in range(local_parms.n_initial_points, local_parms.algo_n_iterations):
            surr_model_state_dict_file = f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/surr_model_{step}.pt"
            if not os.path.exists(surr_model_state_dict_file):
                raise RuntimeError(
                    f"File {surr_model_state_dict_file} does not exist")

            surr_model_state_dicts.append(torch.load(
                surr_model_state_dict_file, map_location=local_parms.device))

        if len(surr_model_state_dicts) != (local_parms.algo_n_iterations - local_parms.n_initial_points):
            raise RuntimeError

        if local_parms.env_discretized:
            embedder = DiscreteEmbbeder(
                num_categories=local_parms.num_categories,
                bounds=local_parms.bounds,
            ).to(device=local_parms.device, dtype=local_parms.torch_dtype)
        else:
            embedder = None

        metrics = compute_metrics(
            env,
            local_parms,
            buffer,
            embedder,
            local_parms.device,
            surr_model_state_dicts=surr_model_state_dicts,
        )
        with open(f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy", "wb") as f:
            np.save(f, metrics)

    # wandb.finish()
