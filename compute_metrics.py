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
from _0_main import Parameters, make_env
from _7_utils import set_seed
from _12_alpine import AlpineN1
from _15_syngp import SynGP
from _16_env_wrapper import EnvWrapper
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from tqdm import tqdm
import os
import argparse
from tensordict import TensorDict
from pathlib import Path
from argparse import ArgumentParser
import wandb

algos_name = [
    "HES-TS-AM-1",
    "HES-TS-AM-10",
    "HES-TS-AM-20",
    "HES-TS-1",
    "HES-TS-2",
    "HES-TS-3",
    "HES-AM-1",
    "HES-AM-2",
    "HES-AM-3",
    "HES-1",
    "HES-2",
    "HES-3",
    "MSL-3",
    "SR",
    "EI",
    "PI",
    "UCB",
    "KG",
]

algos = [
    "HES-TS-AM-1",
    "HES-TS-AM-10",
    "HES-TS-AM-20",
    "HES-TS-1",
    "HES-TS-2",
    "HES-TS-3",
    "HES-AM-1",
    "HES-AM-2",
    "HES-AM-3",
    "HES-1",
    "HES-2",
    "HES-3",
    "qMSL",
    "qSR",
    "qEI",
    "qPI",
    "qUCB",
    "qKG"
]

seeds = [
    2,
    3,
    5,
    7,
    11
]

env_names = [
    "Ackley",
    "Alpine",
    "Beale",
    "Branin",
    "Cosine8",
    "EggHolder",
    "Griewank",
    "Hartmann",
    "HolderTable",
    "Levy",
    "Powell",
    "SixHumpCamel",
    "StyblinskiTang",
    "SynGP"
]

env_noises = [
    0.0,
    0.01,
    0.1,
]

env_discretizeds = [
    False,
    True
]

cost_functions = [
    "euclidean",
    "manhattan",
    "r-spotlight",
    "non-markovian"
]

# Compute metrics
def compute_metrics(
    env,
    parms,
    buffer,
    device,
    WM_state_dicts=None,
):
    metrics = []
    for step in tqdm(range(parms.n_initial_points-1, parms.algo_n_iterations)):
        u_observed = torch.max(buffer["y"][:step+1]).item()

        ######################################################################
        WM = SingleTaskGP(
            buffer["x"][:step+1],
            buffer["y"][:step+1],
            # input_transform=Normalize(
            #     d=parms.x_dim, bounds=parms.bounds.T),
            # outcome_transform=Standardize(1),
        ).to(device)
        if step == parms.algo_n_iterations - 1:
            # Fit GP
            mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
            fit_gpytorch_model(mll)
        else:
            WM.load_state_dict(WM_state_dicts[step - parms.n_initial_points + 1])
        # Set WM to eval mode
        WM.eval()

        # Initialize A consistently across fantasies
        A = torch.empty(
            [parms.n_restarts, parms.n_actions, parms.x_dim],
            device=device
        )
        A = buffer["x"][step].clone().repeat(
            parms.n_restarts, parms.n_actions, 1)
        A = A + torch.randn_like(A) * 0.01
        A.requires_grad = True

        # Initialize optimizer
        optimizer = torch.optim.AdamW([A], lr=0.01)
        loss_fn = parms.loss_function_class(**parms.loss_func_hypers)
        cost_fn = parms.cost_function_class(**parms.cost_func_hypers)

        for i in range(1000):
            ppd = WM(A)
            y_A = ppd.rsample()

            losses = loss_fn(A=A, Y=y_A) + cost_fn(
                prev_X=buffer["x"][step].expand_as(A), current_X=A, previous_cost=0
            )
            # >>> n_fantasy x batch_size

            loss = losses.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (i+1) % 200 == 0:
                print(f"Eval optim round: {i+1}, Loss: {loss.item():.2f}")

        aidx = losses.squeeze(-1).argmin()
        u_posterior = env(A[aidx]).item()

        ######################################################################
        regret = 1 - u_posterior
        if step >= parms.n_initial_points:
            regret += metrics[-1][-1] # Cummulative regret

        metrics.append([u_observed, u_posterior, regret])
        # wandb.log({"u_observed": u_observed, "u_posterior": u_posterior, "c_regret": regret})
        print({"u_observed": u_observed, "u_posterior": u_posterior, "c_regret": regret})

    return np.array(metrics)


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
if __name__ == '__main__':
    wandb.init(project="nonmyopia-metrics")

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
    parser.add_argument("--test_only", type=str2bool, default=False)
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
    if False: # os.path.exists(f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy"):
        metrics = np.load(
            f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy")
    else:
        buffer_file=f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/buffer.pt"
        if not os.path.exists(buffer_file):
            raise RuntimeError(f"File {buffer_file} does not exist")
        buffer=torch.load(buffer_file, map_location=device)
        
        WM_state_dicts=[]
        for step in range(local_parms.n_initial_points, local_parms.algo_n_iterations):
            WM_state_dict_file=f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/world_model_{step}.pt"
            if not os.path.exists(WM_state_dict_file):
                raise RuntimeError(f"File {WM_state_dict_file} does not exist")
                
            WM_state_dicts.append(torch.load(WM_state_dict_file, map_location=device))

        if len(WM_state_dicts) != (local_parms.algo_n_iterations - local_parms.n_initial_points):
            raise RuntimeError

        metrics=compute_metrics(
            env,
            local_parms,
            buffer,
            device,
            WM_state_dicts=WM_state_dicts,
        )
        with open(f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy", "wb") as f:
            np.save(f, metrics)
            
    wandb.finish()