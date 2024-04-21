#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Run the main experiments."""

import os
import copy
import torch
import numpy as np
import dill as pickle

from pathlib import Path
from argparse import ArgumentParser

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
from gpytorch.constraints import Interval

from _1_run import run
from _4_qhes import qCostFunction, qLossFunctionTopK, qCostFunctionEditDistance
from _7_utils import set_seed
from _9_semifuncs import AntBO, nm_AAs
from _11_kernels import TransformedCategorical
from _12_alpine import AlpineN1
from _14_sequence_design_func import SequenceDesignFunction
from _15_syngp import SynGP


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
        self.continue_once = args.continue_once
        self.test_only = args.test_only
        self.seed = args.seed

        if not self.test_only:
            self.save_dir = (
                f"./results/exp_{args.algo}_{args.env_name}_{args.algo_lookahead_steps}"
            )
        else:
            self.save_dir = self.continue_once
            with open(
                os.path.join(self.save_dir, "parameters.json"), "r", encoding="utf-8"
            ) as file_handle:
                text = file_handle.read()
                lines = text.split("\n")
                for line in lines:
                    if "seed" in line:
                        self.seed = int(line.split(": ")[1])

        self.algo = args.algo
        self.algo_ts = args.algo_ts
        self.env_name = args.env_name
        self.env_noise = args.env_noise
        self.n_actions = 1
        self.y_dim = 1
        self.algo_n_iterations = args.algo_n_iterations
        self.algo_lookahead_steps = args.algo_lookahead_steps
        self.func_noise = 0.0
        if args.env_discretized:
            self.env_discretized = True
            self.num_categories = 20
            self.save_dir += "_env_discretized"
        else:
            self.env_discretized = False
            self.num_categories = None

        self.n_samples = 1  # 64
        self.amortized = True if self.algo == "HES" else False
        self.hidden_dim = 32
        self.acq_opt_lr = 0.001 if self.amortized else 1e-3
        self.acq_opt_iter = 500 if self.amortized else 500
        self.n_restarts = 64

        if self.algo == "HES" and self.algo_lookahead_steps > 1:
            self.n_restarts = 16

        elif self.algo == "qMSL" and self.algo_lookahead_steps > 1:
            self.n_restarts = 16
            self.n_samples = 8

        self.kernel = None
        self.cost_spotlight_k = args.cost_spotlight_k
        self.cost_p_norm = args.cost_p_norm
        self.cost_max_noise = args.cost_max_noise
        self.cost_discount = args.cost_discount
        self.cost_discount_threshold = args.cost_discount_threshold

        if self.env_name == "Ackley":
            self.x_dim = 2
            self.bounds = [-32.768, 32.768]
            self.radius = 4.75

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
            self.bounds = [-512, 512]
            self.radius = 75.0

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
            self.bounds = [-10, 10]
            self.radius = 1.5

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

        elif self.env_name == "AntBO":
            self.x_dim = 11
            self.kernel = TransformedCategorical(
                lengthscale_constraint=Interval(0.01, 0.5),
                ard_num_dims=self.x_dim,
            )

        elif self.env_name == "Sequence":
            self.x_dim = 8
            self.radius = 1
            self.kernel = TransformedCategorical()

        else:
            raise NotImplementedError

        # Random select initial points
        self.bounds = np.array(self.bounds)
        if self.bounds.shape[0] < self.x_dim:
            self.bounds = np.tile(self.bounds, [self.x_dim, 1])
        self.n_initial_points = 3**self.x_dim + 1

        ranges = np.linspace(self.bounds[..., 0], self.bounds[..., 1], 4).T
        range_bounds = np.stack((ranges[:, :-1], ranges[:, 1:]), axis=-1)
        cartesian_idxs = np.array(np.meshgrid(*([[0, 1, 2]] * self.x_dim))).T.reshape(
            -1, self.x_dim
        )
        cartesian_rb = range_bounds[list(range(self.x_dim)), cartesian_idxs]

        self.initial_points = np.concatenate(
            (
                np.random.uniform(
                    low=cartesian_rb[..., 0],
                    high=cartesian_rb[..., 1],
                    size=[3**self.x_dim, self.x_dim],
                ),
                np.random.uniform(
                    low=self.bounds[..., 0],
                    high=self.bounds[..., 1],
                    size=[1, self.x_dim],
                ),
            ),
            axis=0,
        )

        self.task = args.task
        self.set_task_parms()

    def set_task_parms(self):
        r"""Set task-specific parameters."""
        if self.task == "topk":
            self.cost_function_class = qCostFunction
            self.cost_func_hypers = dict(
                radius=self.radius,
                k=self.cost_spotlight_k,
                p_norm=self.cost_p_norm,
                max_noise=self.cost_max_noise,
                discount=self.cost_discount,
                discount_threshold=self.cost_discount_threshold,
            )
            self.loss_function_class = qLossFunctionTopK
            self.loss_func_hypers = dict(
                dist_weight=1,
                dist_threshold=0.5,
            )

            if self.algo == "BudgetedBO":
                self.budget = 50
                self.refill_until_lower_bound_is_reached = True
                # self.objective_function = objective_function
                # self.cost_function = cost_function
                # self.objective_cost_function = objective_cost_function

        elif self.task == "minmax":
            self.n_actions = 2
        else:
            raise NotImplementedError

    def __str__(self):
        r"""Return string representation of parameters."""
        output = []
        for k in self.__dict__.keys():
            output.append(f"{k}: {self.__dict__[k]}")
        return "\n".join(output)


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
    elif name == "AntBO":
        assert x_dim == 11, "AntBO only runs on 11-dim X"
        bbox = {
            "tool": "Absolut",
            "antigen": "1ADQ_A",  # 1ADQ_A; 1FBI_X
            # Put path to Absolut (/ABS/PATH/TO/Absolut/)
            "path": "/dfs/user/sttruong/Absolut/bin",
            "process": 8,  # Number of cores
            "startTask": 0,  # start core id
        }

        f_ = AntBO(
            n_categories=np.array([nm_AAs] * x_dim),
            seq_len=x_dim,
            bbox=bbox,
            normalise=False,
        )
    elif name == "chemical":
        with open("examples/semisynthetic.pt", "rb") as file_handle:
            return pickle.load(file_handle)
    elif name == "Sequence":
        f_ = SequenceDesignFunction(dim=x_dim)
    else:
        raise NotImplementedError

    if name != "AntBO":
        f_.bounds[0, :] = torch.tensor(bounds[..., 0], device=f_.bounds.device)
        f_.bounds[1, :] = torch.tensor(bounds[..., 1], device=f_.bounds.device)

    return f_


def make_save_dir(config):
    r"""Create save directory without overwriting directories."""
    if config.test_only:
        return

    init_dir_path = Path(config.save_dir)
    dir_path = Path(str(init_dir_path))

    for i in range(100):
        try:
            dir_path.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            dir_path = Path(str(init_dir_path) + "_" + str(i).zfill(2))

    config.save_dir = str(dir_path)
    print(f"Created save_dir: {config.save_dir}")

    # Save config to save_dir as parameters.json
    config_path = dir_path / "parameters.json"
    with open(str(config_path), "w", encoding="utf-8") as file_handle:
        config_dict = str(config)
        file_handle.write(config_dict)


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[2])
    parser.add_argument("--task", type=str, default="topk")
    parser.add_argument("--env_names", nargs="+", type=str, default=["SynGP"])
    parser.add_argument("--env_noise", type=float, default=0.0)
    parser.add_argument("--env_discretized", action="store_true")
    parser.add_argument("--algos", nargs="+", type=str, default=["HES"])
    parser.add_argument("--algo_ts", action="store_true")
    parser.add_argument("--algo_n_iterations", type=int)
    parser.add_argument("--algo_lookahead_steps", type=int)
    parser.add_argument("--cost_spotlight_k", type=int, default=100)
    parser.add_argument("--cost_p_norm", type=float, default=2.0)
    parser.add_argument("--cost_max_noise", type=float, default=1e-5)
    parser.add_argument("--cost_discount", type=float, default=0.0)
    parser.add_argument("--cost_discount_threshold", type=float, default=-1)
    parser.add_argument("--gpu_id", nargs="+", type=int, default=[0])
    parser.add_argument("--continue_once", type=str, default="")
    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()

    metrics = {}
    list_processes = []
    num_gpus = len(args.gpu_id)
    num_seeds = len(args.seeds)
    num_algos = len(args.algos)
    for e, env_name in enumerate(args.env_names):
        for i, seed in enumerate(args.seeds):
            set_seed(seed)

            for j, algo in enumerate(args.algos):
                # Copy args
                local_args = copy.deepcopy(args)
                local_args.seed = seed
                local_args.gpu_id = args.gpu_id[(i * num_algos + j) % num_gpus]
                local_args.algo = args.algos[j]
                local_args.env_name = env_name

                local_parms = Parameters(local_args)

                # Make save dir
                make_save_dir(local_parms)

                # Init environment
                env = make_env(
                    name=env_name,
                    x_dim=local_parms.x_dim,
                    bounds=local_parms.bounds,
                    noise_std=local_parms.env_noise,
                )
                env = env.to(
                    dtype=local_parms.torch_dtype,
                    device=local_parms.device,
                )

                # Run trials
                real_loss = run(local_parms, env)

                # Assign loss to dictionary of metrics
                metrics[f"eval_metric_{local_args.algo}_{local_parms.seed}"] = real_loss

                pickle.dump(
                    metrics,
                    open(os.path.join(local_parms.save_dir, "metrics.pkl"), "wb"),
                )
