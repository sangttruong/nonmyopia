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
from threading import Thread
from argparse import ArgumentParser

from botorch.test_functions.synthetic import (
    Ackley,
    Beale,
    Branin,
    Hartmann,
    SixHumpCamel,
    Levy,
    StyblinskiTang,
    EggHolder,
    Powell,
    Griewank,
    HolderTable
)
from gpytorch.constraints import Interval

from _1_run_dc import run
from _4_qhes_dc import qCostFunctionSpotlight, qLossFunctionTopK
from _5_evalplot import draw_metric
from _7_utils import kern_exp_quad_noard, sample_mvn, gp_post, unif_random_sample_domain
from _9_semifuncs import AntBO, nm_AAs
from _11_kernels import TransformedCategorical
from _12_alpine import AlpineN1

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
        self.exp_id = args.exp_id
        self.save_dir = f"./results/exp_{self.exp_id:03d}"
        self.torch_dtype = torch.float32

        self.algo = args.algo
        self.env_name = args.env_name
        self.seed = args.seed
        self.n_actions = 1
        self.x_dim = 2
        self.y_dim = 1
        self.bounds = args.bounds
        self.n_iterations = args.n_iterations
        self.lookahead_steps = args.lookahead_steps
        self.func_noise = 0.0
        self.num_categories = 20

        self.n_samples = 64
        self.amortized = True if self.algo == "HES" else False
        self.hidden_dim = 32
        self.acq_opt_lr = 0.001 if self.amortized else 1e-3
        self.acq_opt_iter = 500 if self.amortized else 500
        self.n_restarts = 64
        if self.algo == "HES" and self.lookahead_steps > 1:
            self.n_restarts = 16
        elif self.algo == "qMSL" and self.lookahead_steps > 1:
            self.n_restarts = 16
            self.n_samples = 8
        
        if self.env_name == "AntBO":
            self.x_dim = 11
            self.kernel = TransformedCategorical(
                lengthscale_constraint=Interval(0.01, 0.5), 
                ard_num_dims=self.x_dim, 
            )
        else:
            # Using default MaternKernel
            self.kernel = None
            
        if self.env_name == "SynGP":
            self.radius = 0.15
            self.initial_points = [
                [0.2, 0.7], [0.0, -0.4], [0.45, 0.5],
                
                [0.4, 0.4],[0.4, 0.35],[0.4, 0.2],
                [0.4, 0.1],[0.4, 0.0],[0.35, -0.1],
                [0.2, -0.2],[0.2, -0.2],[0.1, -0.3],
                [0.1, -0.4],[0.1, -0.5],[0.0, -0.5],
                [-0.1, -0.55],[-0.2, -0.55],[-0.1, -0.5],
            ]
            
        elif self.env_name == "HolderTable":
            self.radius = 0.75
            self.initial_points = [
                [7.0, 9.0], [8.0, 5.4], [7.0, 4.2],
                [2.0, 4.5], [7.0, 2.0], [4.0, 8.3], 
                [2.0, 3.0], [6.5, 1.0], [5.0, 5.0],
            ]
            
        elif self.env_name == "EggHolder":
            self.radius = 80.0
            self.initial_points = [
                [-300.0, 400.0], [-200.0, 0.0], [200, -200],
                [300.0, 0.0], [100.0, 300.0], [0.0, 0.0],
            ]
            
        elif self.env_name == "Alpine":
            self.radius = 0.8
            self.initial_points = [
                [7.5, 7.5], [5.0, 5.0], [4.0, 7.0],
                [7.0, 3.2], [2.8, 5], [5.0, 4.0]
            ]
        else:
            raise NotImplementedError
        
        self.n_initial_points = len(self.initial_points)
        self.task = args.task
        self.set_task_parms()
            
    def set_task_parms(self):
        r"""Set task-specific parameters."""
        if self.task == "topk":
            self.cost_function_class = qCostFunctionSpotlight
            self.cost_func_hypers = dict(radius=self.radius)
            self.loss_function_class = qLossFunctionTopK
            self.loss_func_hypers = dict(
                dist_weight=1,
                dist_threshold=0.5,
            )
                
            if self.algo == "BudgetedBO":
                self.budget = 50
                self.refill_until_lower_bound_is_reached = True
                self.objective_function=objective_function
                self.cost_function=cost_function
                self.objective_cost_function=objective_cost_function
                
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


class SynGP:
    """Synthetic functions defined by draws from a Gaussian process."""

    def __init__(self, dim, seed=8):
        self.bounds = torch.tensor([[-1, 1]] * dim).T
        self.seed = seed
        self.n_obs = 10
        self.hypers = {"ls": 0.25, "alpha": 1.0, "sigma": 1e-2, "n_dimx": dim}
        self.domain_samples = None
        self.prior_samples = None

    def initialize(self):
        """Initialize synthetic function."""
        self.set_random_seed()
        self.set_kernel()
        self.draw_domain_samples()
        self.draw_prior_samples()

    def set_random_seed(self):
        """Set random seed."""
        np.random.seed(self.seed)

    def set_kernel(self):
        """Set self.kernel function."""

        def kernel(xlist1, xlist2, ls, alpha):
            return kern_exp_quad_noard(xlist1, xlist2, ls, alpha)

        self.kernel = kernel

    def draw_domain_samples(self):
        """Draw uniform random samples from self.domain."""
        domain_samples = unif_random_sample_domain(self.bounds.T, self.n_obs)
        self.domain_samples = np.array(domain_samples).reshape(self.n_obs, -1)

    def draw_prior_samples(self):
        """Draw a prior function and evaluate it at self.domain_samples."""
        domain_samples = self.domain_samples
        prior_mean = np.zeros(domain_samples.shape[0])
        prior_cov = self.kernel(
            domain_samples, domain_samples, self.hypers["ls"], self.hypers["alpha"]
        )
        prior_samples = sample_mvn(prior_mean, prior_cov, 1)
        self.prior_samples = prior_samples.reshape(self.n_obs, -1)

    def __call__(self, test_x):
        """
        Call synthetic function on test_x, and return the posterior mean given by
        self.get_post_mean method.
        """
        if self.domain_samples is None or self.prior_samples is None:
            self.initialize()

        test_x = self.process_function_input(test_x)
        post_mean = self.get_post_mean(test_x)
        test_y = self.process_function_output(post_mean)

        return test_y

    def get_post_mean(self, test_x):
        """
        Return mean of model posterior (given self.domain_samples, self.prior_samples)
        at the test_x inputs.
        """
        post_mean, _ = gp_post(
            self.domain_samples,
            self.prior_samples,
            test_x,
            self.hypers["ls"],
            self.hypers["alpha"],
            self.hypers["sigma"],
            self.kernel,
        )
        return post_mean

    def process_function_input(self, test_x):
        """Process and possibly reshape inputs to the synthetic function."""
        self.device = test_x.device
        test_x = test_x.cpu().detach().numpy()
        if len(test_x.shape) == 1:
            test_x = test_x.reshape(1, -1)
            self.input_mode = "single"
        elif len(test_x.shape) == 0:
            assert self.hypers["n_dimx"] == 1
            test_x = test_x.reshape(1, -1)
            self.input_mode = "single"
        else:
            self.input_mode = "batch"

        return test_x

    def process_function_output(self, func_output):
        """Process and possibly reshape output of the synthetic function."""
        if self.input_mode == "single":
            func_output = func_output[0][0]
        elif self.input_mode == "batch":
            func_output = func_output.reshape(-1, 1)

        return torch.tensor(func_output, dtype=self.dtype, device=self.device)

    def to(self, dtype, device):
        self.dtype = dtype
        return self


def make_env(env_name, x_dim, bounds):
    r"""Make environment."""
    if env_name == "Ackley":
        f_ = Ackley(dim=x_dim, negate=True)
    elif env_name == "Beale":
        f_ = Beale(negate=False)
    elif env_name == "Branin":
        f_ = Branin(dim=x_dim, negate=False)
    elif env_name == "SixHumpCamel":
        f_ = SixHumpCamel(dim=x_dim, negate=False)
    elif env_name == "Hartmann":
        f_ = Hartmann(dim=x_dim, negate=False)
    elif env_name == "Levy":
        f_ = Levy(dim=x_dim, negate=True)
    elif env_name == "StyblinskiTang":
        f_ = StyblinskiTang(dim=x_dim, negate=True)
    elif env_name == "EggHolder":
        f_ = EggHolder(negate=False)
    elif env_name == "Powell":
        f_ = Powell(dim=x_dim, negate=True)
    elif env_name == "Griewank":
        f_ = Griewank(dim=2)
    elif env_name == "HolderTable":
        f_ = HolderTable(negate=True)
    elif env_name == "Alpine":
        f_ = AlpineN1(dim=x_dim)
    elif env_name == "SynGP":
        f_ = SynGP(dim=x_dim)
    elif env_name == "AntBO":
        assert x_dim == 11, "AntBO only runs on 11-dim X"
        bbox = {
            "tool": "Absolut",
            "antigen": "1ADQ_A", # 1ADQ_A; 1FBI_X
            "path": "/dfs/user/sttruong/Absolut/bin",  # Put path to Absolut (/ABS/PATH/TO/Absolut/)
            "process": 8,  # Number of cores
            "startTask": 0,  # start core id
        }

        f_ = AntBO(
            n_categories=np.array([nm_AAs] * x_dim),
            seq_len=x_dim,
            bbox=bbox,
            normalise=False,
        )
    elif env_name == "chemical":
        with open("examples/semisynthetic.pt", "rb") as file_handle:
            return pickle.load(file_handle)
    else:
        raise NotImplementedError

    if env_name != "AntBO":
        f_.bounds[0, :].fill_(bounds[0])
        f_.bounds[1, :].fill_(bounds[1])
        
    return f_


def make_save_dir(config):
    r"""Create save directory without overwriting directories."""
    init_dir_path = Path(config.save_dir)
    dir_path = Path(str(init_dir_path))

    for i in range(50):
        try:
            dir_path.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            dir_path = Path(str(init_dir_path) + "_" + str(i).zfill(2))

    config.save_dir = str(dir_path)
    print(f"Created save_dir: {config.save_dir}")

    # Save config to save_dir as parameters.json
    config_path = dir_path / "parameters.json"
    with open(str(config_path), "w") as file_handle:
        config_dict = str(config)
        file_handle.write(config_dict)


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[2])
    parser.add_argument("--task", type=str, default="topk")
    parser.add_argument("--env_name", type=str, default="Ackley")
    parser.add_argument("--bounds", nargs="+", type=float, default=[-1, 1])
    parser.add_argument("--algos", nargs="+", type=str, default=["HES"])
    parser.add_argument("--n_iterations", type=int)
    parser.add_argument("--lookahead_steps", type=int)
    parser.add_argument("--exp_id", type=int, default=0)
    parser.add_argument("--gpu_id", nargs="+", type=int)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    metrics = {}
    list_processes = []
    num_gpus = len(args.gpu_id)
    num_seeds = len(args.seeds)
    num_algos = len(args.algos)
    for i, seed in enumerate(args.seeds):
        for j, algo in enumerate(args.algos):
            # Copy args
            local_args = copy.deepcopy(args)
            local_args.seed = seed
            local_args.gpu_id = args.gpu_id[(i * num_algos + j) % num_gpus]
            local_args.algo = args.algos[j]

            local_parms = Parameters(local_args)

            # Make save dir
            make_save_dir(local_parms)

            # Init environment
            env = make_env(
                env_name=local_parms.env_name,
                x_dim=local_parms.x_dim,
                bounds=local_parms.bounds
            )
            env = env.to(
                dtype=local_parms.torch_dtype,
                device=local_parms.device,
            )

            # Run trials
            real_loss = run(local_parms, env)

            # Assign loss to dictionary of metrics
            metrics[f"eval_metric_{local_args.algo}_{local_args.seed}"] = real_loss

            import pickle

            pickle.dump(
                metrics, open(os.path.join(local_parms.save_dir, "metrics.pkl"), "wb")
            )

            # p = Thread(
            #     target=run,
            #     args=(local_parms, env, metrics),
            # )
            # list_processes.append(p)

    # # Implement a simple queue system to run the experiments
    # number_alive_processes = 0
    # list_alive_processes = []
    # for i, p in enumerate(list_processes):
    #     p.start()
    #     list_alive_processes.append(i)
    #     if len(list_alive_processes) >= args.n_jobs:
    #         while True:
    #             for j in list_alive_processes:
    #                 if not list_processes[j].is_alive():
    #                     list_alive_processes.remove(j)

    #             if len(list_alive_processes) < args.n_jobs:
    #                 break

    #             time.sleep(0.5)

    # for pi in list_alive_processes:
    #     list_processes[pi].join()

    # Draw regret curves
    list_metrics = []
    for i, algo in enumerate(args.algos):
        algo_metrics = []
        for i, seed in enumerate(args.seeds):
            algo_metrics.append(metrics[f"eval_metric_{algo}_{seed}"])
        list_metrics.append(algo_metrics)

    draw_metric("results", list_metrics, args.algos)
