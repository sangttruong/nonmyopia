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
# from synthetic_functions.alpine import AlpineN1
# from synthetic_functions.syngp import SynGP
from env_wrapper import EnvWrapper
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
from tueplots import bundles
plt.rcParams.update(bundles.neurips2024())

algos_name = [
    # "HES-TS-AM-1",
    # "HES-TS-AM-10",
    "HES-TS-AM-20",
    # "HES-TS-1",
    # "HES-TS-2",
    # "HES-TS-3",
    # "HES-AM-1",
    # "HES-AM-2",
    # "HES-AM-3",
    # "HES-1",
    # "HES-2",
    # "HES-3",
    "MSL-3",
    "SR",
    "EI",
    "PI",
    "UCB",
    "KG"
]

algos = [
    # "HES-TS-AM-1",
    # "HES-TS-AM-10",
    "HES-TS-AM-20",
    # "HES-TS-1",
    # "HES-TS-2",
    # "HES-TS-3",
    # "HES-AM-1",
    # "HES-AM-2",
    # "HES-AM-3",
    # "HES-1",
    # "HES-2",
    # "HES-3",
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
    # "Alpine",
    # "Beale",
    # "Branin",
    # "Cosine8",
    # "EggHolder",
    # "Griewank",
    # "Hartmann",
    # "HolderTable",
    # "Levy",
    # "Powell",
    # "SixHumpCamel",
    # "StyblinskiTang",
    # "SynGP"
]

env_noises = [
    0.0,
    # 0.01,
    # 0.1,
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

cost_function_names = {
    "euclidean": "Euclidean cost",
    "manhattan": "Manhattan cost",
    "r-spotlight": "$r$-spotlight cost",
    "non-markovian": "Non-Markovian cost"
}

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

# Draw metrics
def draw_metric(
    metric_names,
    env_name,
    dict_metrics,
    save_file,
):
    fig, axs = plt.subplots(1, len(metric_names),
                            figsize=(4 * len(metric_names), 3))
    for mid, metric_name in enumerate(metric_names):
        if len(metric_names) == 1:
            ax = axs
        else:
            ax = axs[mid]
        for algo, metrics in dict_metrics.items():
            mean = np.mean(metrics, axis=0)
            std = np.std(metrics, axis=0)
            
            mean_line = ax.plot(np.arange(mean.shape[0]), mean[:, mid], label=algo)
            ax.fill_between(
                np.arange(mean.shape[0]), mean[:, mid] - std[:, mid], mean[:, mid] + std[:, mid], alpha=0.1
            )
            
        ax.set_xlabel("Step")
        ax.set_ylabel(metric_name)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, bbox_to_anchor=(0.5, -0.15))
    fig.suptitle(f"{env_name}", fontsize=11)
    plt.savefig(save_file, dpi=300)
    plt.close()

def draw_time(
    env_name,
    dict_metrics,
    save_file,
):
    fig = plt.figure(figsize=(4, 3))
    
    for idx, (algo, metrics) in enumerate(dict_metrics.items()):
        mean = np.mean(metrics)
        std = np.std(metrics)
        mean_line = plt.bar(algo, mean, yerr=std, label=algo, fill=False, capsize=5, error_kw={"elinewidth": 0.75, "markeredgewidth": 0.75})
        
    plt.ylabel("Time (s)")
    plt.title(f"{env_name}", fontsize=11)
    plt.savefig(save_file, dpi=300)
    plt.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

print("Drawing runtime...")
for env_discretized in env_discretizeds:
    dict_metrics = {}
    for env_name in env_names:
        x_dim, bounds, radius, n_initial_points, algo_n_iterations=get_env_info(
            env_name, device)
        for env_noise in env_noises:
            for cost_fn in cost_functions:
                for aid, algo in enumerate(algos):
                    list_metrics = []
                    for seed in seeds:
                        buffer_file=f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/buffer.pt"
                        if not os.path.exists(buffer_file) and not os.path.exists(f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy"):
                            continue
                        try:
                            buffer=torch.load(buffer_file, map_location=device)
                        except:
                            continue
                        # >>> n_iterations x 1
                        list_metrics.append(buffer["runtime"][n_initial_points:].cpu().unsqueeze(-1).tolist())
        
                    if len(list_metrics) == 0:
                        continue
                    # >>> n_seeds x n_iterations x 1
    
                    list_metrics = np.array(list_metrics).mean(keepdims=True)
                    if np.isnan(np.sum(list_metrics)):
                        continue

                    if algos_name[aid] not in dict_metrics:
                        dict_metrics[algos_name[aid]] = list_metrics
                    else:
                        dict_metrics[algos_name[aid]] = np.concatenate([dict_metrics[algos_name[aid]],list_metrics], axis=0)
            
    if len(dict_metrics) == 0:
        continue
        
    save_dir = Path(
        f"plots/runtime")
    save_dir.mkdir(parents=True, exist_ok=True)
    draw_time(
        "",
        dict_metrics,
        f"plots/runtime/runtime{'_discretized' if env_discretized else ''}.png",
    )
        

# Create all triplet (env_name, env_noise, env_discretized)
datasets = []
for env_name in env_names:
    for env_noise in env_noises:
        for env_discretized in env_discretizeds:
            datasets.append((env_name, env_noise, env_discretized))
            
for dataset in datasets:
    print("Drawing for dataset", dataset)
    env_name, env_noise, env_discretized = dataset
    
    for cost_fn in cost_functions:
        dict_metrics = {}
        for aid, algo in enumerate(algos):
            list_metrics = []
            for seed in seeds:
                if os.path.exists(f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy"):
                    metrics = np.load(
                        f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy")
                else:
                    continue
                    
                # >>> n_iterations x 3
                list_metrics.append(metrics)

            if len(list_metrics) == 0:
                continue
            # >>> n_seeds x n_iterations x 3
            
            dict_metrics[algos_name[aid]] = np.array(list_metrics)[..., -1:] # Get the Regret value

        if len(dict_metrics) == 0:
            continue

        # >>> algo x n_seeds x n_iterations x 3
        save_dir = Path(
            f"plots/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}")
        save_dir.mkdir(parents=True, exist_ok=True)
        draw_metric(
            # ["$u_{observed}$", "$u_{posterior}$", "Regret"],
            ["Regret"],
            f"{'Discretized ' if env_discretized else ''}{env_name} with $\sigma=${env_noise}\% and {cost_function_names[cost_fn]}",
            dict_metrics,
            f"plots/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{cost_fn}.png",
        )
