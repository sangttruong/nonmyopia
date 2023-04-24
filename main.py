#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Run the main experiments."""

import copy
from pathlib import Path
import torch

from threading import Thread
from argparse import ArgumentParser
import dill as pickle
from botorch.test_functions.synthetic import Ackley

from run import run
from experiment.checkpoint_manager import make_save_dir
from utils.plot import draw_metric, draw_posterior
from models.EHIG import qCostFunctionSpotlight, qLossFunctionTopK


class Parameters:
    def __init__(self, args):
        # general arguments
        self.task = args.task
        self.set_task_parms()

        self.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        print(f"Using device {self.device}")
        self.gpu_id = args.gpu_id
        self.exp_id = args.exp_id
        self.mode = "train"
        self.check_dir = "experiments"
        self.save_dir = f"./results/exp_{self.exp_id:03d}"
        self.torch_dtype = torch.double

        self.algo = args.algo
        self.env_name = args.env_name

        self.seed = args.seed
        self.seed_synthfunc = 1
        self.x_dim = 2
        self.y_dim = 1
        self.bounds = [-1, 1]
        self.n_iterations = 10
        self.lookahead_steps = 1
        self.n_initial_points = 10
        self.local_init = True
        self.n_candidates = 1
        self.func_is_noisy = False
        self.func_noise = 0.1
        self.plot_iters = list(range(0, 101, 1))
        self.start_iter = 0

        # Algorithm parameters
        self.batch_size = 10
        self.lookahead_batch_sizes = [2] * self.lookahead_steps
        self.num_fantasies = [2] * self.lookahead_steps

        if self.algo == "HES":
            self.use_amortized_optimization = True
        else:
            self.use_amortized_optimization = False

        self.acq_opt_lr = 0.05 if self.use_amortized_optimization else 1e-3
        self.n_samples = 64
        self.decay_factor = 1

        # Optimizer
        # self.optimizer = "adam"
        self.acq_opt_iter = 1000 if self.use_amortized_optimization else 1000
        # self.acq_opt_iter = 500 if self.use_amortized_optimization else 3000
        self.acq_warmup_iter = self.acq_opt_iter // 20
        self.acq_earlystop_iter = int(self.acq_opt_iter * 0.4)
        self.n_restarts = 1

        # Amortization
        self.hidden_dim = 128
        self.n_layers = 2
        self.activation = "elu"
        self.hidden_coeff = 4

        self.init_noise_thredhold = 0.01

        # Resampling
        """When n_resampling_max == 1 and n_resampling_improvement_threadhold is small, we have 
        the orange curve. n_resampling_max is large and n_resampling_improvement_threadhold is
        large, we have the pink curve (closer to stochastic gradient descent). We can interpolate
        between these 2 options by setting both hyperparameters to some moderate value. """
        self.n_resampling_max = 1
        self.n_resampling_improvement_threadhold = 0.01

        # Patients
        self.max_patient = 5000
        self.max_patient_resampling = 5

        # annealing for hes optimizer
        """When eta_min = acq_opt_lr, the learning rate is constant at acq_opt_lr
        large T_max corresponds to slow annealing
        """
        self.eta_min = 0.0001
        self.T_max = 100

    def set_task_parms(self):
        if self.task == "topk":
            self.eval_function = draw_posterior  # TODO
            self.final_eval_function = None  # TODO
            self.plot_function = None  # TODO
            self.n_actions = 1
            self.epsilon = 1  # 1: no random reset, 0: random reset

            self.loss_function_class = qLossFunctionTopK
            self.loss_function_hyperparameters = dict(
                dist_weight=1,
                dist_threshold=0.5,
            )
            self.cost_function_class = qCostFunctionSpotlight
            self.cost_function_hyperparameters = dict(radius=0.1)

        elif self.task == "minmax":
            self.eval_function = None  # TODO
            self.final_eval_function = None  # TODO
            self.plot_function = None  # TODO
            self.n_actions = 2

        elif self.task == "twovalue":
            self.eval_function = None  # TODO
            self.final_eval_function = None  # TODO
            self.plot_function = None  # TODO
            self.n_actions = None  # TODO

        elif self.task == "mvs":
            self.eval_function = None  # TODO
            self.final_eval_function = None  # TODO
            self.plot_function = None  # TODO
            self.n_actions = None  # TODO

        elif self.task == "levelset":
            self.eval_function = None  # TODO
            self.final_eval_function = None  # TODO
            self.plot_function = None  # TODO
            self.n_actions = None  # TODO

        elif self.task == "multilevelset":
            self.eval_function = None  # TODO
            self.final_eval_function = None  # TODO
            self.plot_function = None  # TODO
            self.n_actions = None  # TODO

        elif self.task == "pbest":
            self.eval_function = None  # TODO
            self.final_eval_function = None  # TODO
            self.plot_function = None  # TODO
            self.n_actions = None  # TODO

        elif self.task == "bestofk":
            self.eval_function = None  # TODO
            self.final_eval_function = None  # TODO
            self.plot_function = None  # TODO
            self.n_actions = None  # TODO

        else:
            raise NotImplementedError

    def __str__(self):
        output = []
        for k in self.__dict__.keys():
            output.append(f"{k}: {self.__dict__[k]}")
        return "\n".join(output)


def make_env(env_name, x_dim, bounds):
    if env_name == "Ackley":
        f_ = Ackley(dim=x_dim, negate=False)
        f_.bounds[0, :].fill_(bounds[0])
        f_.bounds[1, :].fill_(bounds[1])
        return f_

    elif env_name == "chemical":
        with open("examples/semisynthetic.pt", "rb") as file_handle:
            return pickle.load(file_handle)
    else:
        raise NotImplementedError

def make_save_dir(config):
    """Create save directory safely (without overwriting directories), using config."""
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[2, 3])
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
            run(local_parms, env, metrics)

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

    # Convert the metrics to a normal dict
    metrics = dict(metrics)

    # Draw regret curves
    list_metrics = []
    for i, algo in enumerate(args.algos):
        algo_metrics = []
        for i, seed in enumerate(args.seeds):
            algo_metrics.append(metrics[f"eval_metric_{algo}_{seed}"])
        list_metrics.append(algo_metrics)

    draw_metric("results", list_metrics, args.algos)
