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

# Init n_initial_points and algo_n_iterations


def get_env_info(env_name, device):
    if env_name == "Ackley":
        x_dim = 2
        bounds = [-2, 2]
        radius = 0.3
        n_initial_points = 50
        algo_n_iterations = 100

    elif env_name == "Alpine":
        x_dim = 2
        bounds = [0, 10]
        radius = 0.75
        n_initial_points = 100
        algo_n_iterations = 150

    elif env_name == "Beale":
        x_dim = 2
        bounds = [-4.5, 4.5]
        radius = 0.65
        n_initial_points = 100
        algo_n_iterations = 150

    elif env_name == "Branin":
        x_dim = 2
        bounds = [[-5, 10], [0, 15]]
        radius = 1.2
        n_initial_points = 20
        algo_n_iterations = 70

    elif env_name == "Cosine8":
        x_dim = 8
        bounds = [-1, 1]
        radius = 0.3
        n_initial_points = 200
        algo_n_iterations = 300

    elif env_name == "EggHolder":
        x_dim = 2
        bounds = [-100, 100]
        radius = 15.0
        n_initial_points = 100
        algo_n_iterations = 150

    elif env_name == "Griewank":
        x_dim = 2
        bounds = [-600, 600]
        radius = 85.0
        n_initial_points = 20
        algo_n_iterations = 70

    elif env_name == "Hartmann":
        x_dim = 6
        bounds = [0, 1]
        radius = 0.15
        n_initial_points = 500
        algo_n_iterations = 600

    elif env_name == "HolderTable":
        x_dim = 2
        bounds = [0, 10]
        radius = 0.75
        n_initial_points = 100
        algo_n_iterations = 150

    elif env_name == "Levy":
        x_dim = 2
        bounds = [-10, 10]
        radius = 1.5
        n_initial_points = 75
        algo_n_iterations = 125

    elif env_name == "Powell":
        x_dim = 4
        bounds = [-4, 5]
        radius = 0.9
        n_initial_points = 100
        algo_n_iterations = 200

    elif env_name == "SixHumpCamel":
        x_dim = 2
        bounds = [[-3, 3], [-2, 2]]
        radius = 0.4
        n_initial_points = 40
        algo_n_iterations = 90

    elif env_name == "StyblinskiTang":
        x_dim = 2
        bounds = [-5, 5]
        radius = 0.75
        n_initial_points = 45
        algo_n_iterations = 95

    elif env_name == "SynGP":
        x_dim = 2
        bounds = [-1, 1]
        radius = 0.15
        n_initial_points = 25
        algo_n_iterations = 75

    else:
        raise NotImplementedError

    bounds = np.array(bounds)
    if bounds.ndim < 2 or bounds.shape[0] < x_dim:
        bounds = np.tile(bounds, [x_dim, 1])
    bounds = torch.tensor(bounds, dtype=torch.double, device=device)

    return x_dim, bounds, radius, n_initial_points, algo_n_iterations


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
        f_ = Griewank(dim=x_dim, noise_std=noise_std)
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

    if name != "AntBO":
        f_.bounds[0, :] = bounds[..., 0]
        f_.bounds[1, :] = bounds[..., 1]

    return EnvWrapper(name, f_)

# Compute metrics


def compute_metrics(
    env,
    env_info,
    buffer,
    device,
    WM_state_dicts=None,
):
    x_dim, bounds, radius, n_initial_points, algo_n_iterations = env_info

    metrics = []
    for step in tqdm(range(n_initial_points, algo_n_iterations)):
        u_observed = torch.max(buffer["y"][:step]).item()

        ######################################################################
        WM = SingleTaskGP(
            buffer["x"][:step],
            buffer["y"][:step],
            input_transform=Normalize(
                d=x_dim, bounds=bounds.T),
            outcome_transform=Standardize(1),
        ).to(device)
        # mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
        # fit_gpytorch_model(mll)
        # if WM_state_dicts is not None:
        WM.load_state_dict(WM_state_dicts[step - n_initial_points])

        def _max_fn_():
            # Sample 10000 points and find the maximum
            inputs = torch.rand((10000, x_dim)).to(device)
            inputs = inputs * (bounds[..., 1] - bounds[..., 0]) + bounds[..., 0]
            res = WM.posterior(inputs).mean
            _max = torch.max(res, dim=0)
            max_y = _max.values.item()
            max_X = inputs[_max.indices.item()]
            return max_X, max_y

        max_X, max_y = _max_fn_()
        for _ in range(9):
            _max_X, _max_y = _max_fn_()
            if _max_y > max_y:
                max_X, max_y = _max_X, _max_y

        u_posterior = env(max_X).item()

        ######################################################################
        regret = 1 - u_posterior

        metrics.append([u_observed, u_posterior, regret])
        wandb.log({"u_observed": u_observed, "u_posterior": u_posterior, "regret": regret})

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
    parser.add_argument("--env_name", type=str, default="SynGP")
    parser.add_argument("--env_noise", type=float, default=0.0)
    parser.add_argument("--env_discretized", type=str2bool, default=False)
    parser.add_argument("--algo", type=str, default="HES-TS-AM-20")
    parser.add_argument("--cost_fn", type=str, default="euclidean")
    args = parser.parse_args()

    seed = args.seed
    env_name = args.env_name
    env_noise = args.env_noise
    env_discretized = args.env_discretized
    algo = args.algo
    cost_fn = args.cost_fn
    
    x_dim, bounds, radius, n_initial_points, algo_n_iterations=get_env_info(
        env_name, device)
    noise_std=env_noise * np.max((bounds[..., 1] - bounds[..., 0]).float().cpu().numpy()) / 100
    env=make_env(
        env_name,
        x_dim,
        bounds,
        noise_std=noise_std
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
        for step in range(n_initial_points, algo_n_iterations):
            WM_state_dict_file=f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/world_model_{step}.pt"
            if not os.path.exists(WM_state_dict_file):
                raise RuntimeError(f"File {WM_state_dict_file} does not exist")
                
            WM_state_dicts.append(torch.load(WM_state_dict_file, map_location=device))

        if len(WM_state_dicts) != (algo_n_iterations - n_initial_points):
            raise RuntimeError

        metrics=compute_metrics(
            env,
            (x_dim, bounds, radius, n_initial_points, algo_n_iterations),
            buffer,
            device,
            WM_state_dicts=WM_state_dicts,
        )
        with open(f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy", "wb") as f:
            np.save(f, metrics)
            
    wandb.finish()