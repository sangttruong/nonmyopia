#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Run the main experiments."""

import copy
from argparse import ArgumentParser
from pathlib import Path

from threading import Thread

import dill as pickle
import torch
from botorch.test_functions.synthetic import Ackley, Beale, Branin, Hartmann
from _1_run import run
from _4_qhes import qCostFunctionSpotlight, qLossFunctionTopK
from _5_evalplot import draw_metric


class Parameters:
    r"""Class to store all parameters for the experiment."""

    def __init__(self, args):
        r"""Initialize parameters."""
        # general parameters
        self.task = args.task
        self.set_task_parms()

        if torch.cuda.is_available():
            self.device = f"cuda:{args.gpu_id}"
        else:
            self.device = "cpu"
        self.gpu_id = args.gpu_id
        self.exp_id = args.exp_id
        self.save_dir = f"./results/exp_{self.exp_id:03d}"
        self.torch_dtype = torch.float32

        self.algo = args.algo
        self.env_name = args.env_name
        self.seed = args.seed
        self.x_dim = 2
        self.y_dim = 1
        self.bounds = [-1, 1]
        self.n_iterations = 20
        self.lookahead_steps = 20
        self.n_initial_points = 2
        self.func_noise = 0.0

        self.n_samples = 12
        self.amortized = True if self.algo == "HES" else False
        self.hidden_dim = 32
        self.acq_opt_lr = 0.0001 if self.amortized else 1e-3
        self.acq_opt_iter = 400 if self.amortized else 1000
        self.n_restarts = 64

    def set_task_parms(self):
        r"""Set task-specific parameters."""
        if self.task == "topk":
            self.n_actions = 1
            self.loss_function_class = qLossFunctionTopK
            self.loss_func_hypers = dict(
                dist_weight=1,
                dist_threshold=0.5,
            )
            self.cost_function_class = qCostFunctionSpotlight
            self.cost_func_hypers = dict(radius=0.25)
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


def make_env(env_name, x_dim, bounds):
    r"""Make environment."""
    if env_name == "Ackley":
        f_ = Ackley(dim=x_dim, negate=False)
    elif env_name == "Beale":
        f_ = Beale(negate=False)
    elif env_name == "chemical":
        with open("examples/semisynthetic.pt", "rb") as file_handle:
            return pickle.load(file_handle)
    else:
        raise NotImplementedError

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
    parser.add_argument("--algos", nargs="+", type=str, default=["HES"])
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
                bounds=local_parms.bounds,
            )
            env = env.to(
                dtype=local_parms.torch_dtype,
                device=local_parms.device,
            )

            # Run trials
            real_loss = run(local_parms, env)

            # Assign loss to dictionary of metrics
            metrics[f"eval_metric_{local_args.algo}_{local_args.seed}"] = real_loss

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
