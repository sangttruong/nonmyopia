#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Run the main experiments."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
from acqfs import qCostFunction, qLossFunctionTopK
from actor import Actor
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from env_embedder import DiscreteEmbbeder
from gpytorch.mlls import ExactMarginalLogLikelihood
from tensordict import TensorDict
from utils import make_env, make_save_dir, set_seed, str2bool


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
        self.torch_dtype = torch.double
        self.cont = args.cont
        self.seed = args.seed
        self.plot = args.plot

        self.algo = args.algo
        self.algo_ts = False
        self.env_name = args.env_name
        self.n_actions = 1
        self.y_dim = 1
        self.algo_n_iterations = None

        self.n_samples = 64  # 1
        self.amortized = False
        self.hidden_dim = 32
        self.acq_opt_lr = 0.001 if self.amortized else 1e-3
        self.acq_opt_iter = 500 if self.amortized else 500
        self.n_restarts = 64

        if self.algo.startswith("HES"):
            self.n_restarts = 16
            self.algo = "HES"
            self.algo_lookahead_steps = int(args.algo.split("-")[-1])
            self.algo_ts = "TS" in args.algo
            self.amortized = "AM" in args.algo
        elif self.algo == "qMSL":
            self.n_restarts = 4
            self.n_samples = 4
            self.algo_lookahead_steps = 3  # Equivalent 4 in HES
        elif self.algo == "qKG":
            self.algo_lookahead_steps = 1
        else:
            self.algo_lookahead_steps = 0

        self.kernel = None
        if self.env_name == "Ackley":
            self.x_dim = 2
            self.bounds = [-2, 2]
            self.n_initial_points = 50
            self.algo_n_iterations = 100

        elif self.env_name == "Alpine":
            self.x_dim = 2
            self.bounds = [0, 10]
            self.n_initial_points = 100
            self.algo_n_iterations = 150

        elif self.env_name == "Beale":
            self.x_dim = 2
            self.bounds = [-4.5, 4.5]
            self.n_initial_points = 100
            self.algo_n_iterations = 150

        elif self.env_name == "Branin":
            self.x_dim = 2
            self.bounds = [[-5, 10], [0, 15]]
            self.n_initial_points = 20
            self.algo_n_iterations = 70

        elif self.env_name == "Cosine8":
            self.x_dim = 8
            self.bounds = [-1, 1]
            self.n_initial_points = 200
            self.algo_n_iterations = 300

        elif self.env_name == "EggHolder":
            self.x_dim = 2
            self.bounds = [-100, 100]
            self.n_initial_points = 100
            self.algo_n_iterations = 150

        elif self.env_name == "Griewank":
            self.x_dim = 2
            self.bounds = [-600, 600]
            self.n_initial_points = 20
            self.algo_n_iterations = 70

        elif self.env_name == "Hartmann":
            self.x_dim = 6
            self.bounds = [0, 1]
            self.n_initial_points = 500
            self.algo_n_iterations = 600

        elif self.env_name == "HolderTable":
            self.x_dim = 2
            self.bounds = [0, 10]
            self.n_initial_points = 100
            self.algo_n_iterations = 150

        elif self.env_name == "Levy":
            self.x_dim = 2
            self.bounds = [-10, 10]
            self.n_initial_points = 75
            self.algo_n_iterations = 125

        elif self.env_name == "Powell":
            self.x_dim = 4
            self.bounds = [-4, 5]
            self.n_initial_points = 100
            self.algo_n_iterations = 200

        elif self.env_name == "SixHumpCamel":
            self.x_dim = 2
            self.bounds = [[-3, 3], [-2, 2]]
            self.n_initial_points = 40
            self.algo_n_iterations = 90

        elif self.env_name == "StyblinskiTang":
            self.x_dim = 2
            self.bounds = [-5, 5]
            self.n_initial_points = 45
            self.algo_n_iterations = 95

        elif self.env_name == "SynGP":
            self.x_dim = 2
            self.bounds = [-1, 1]
            self.n_initial_points = 25
            self.algo_n_iterations = 75

        else:
            raise NotImplementedError

        if self.x_dim == 2:
            self.radius = 0.075
        elif self.x_dim == 4:
            self.radius = 0.1
        elif self.x_dim == 6:
            self.radius = 0.125
        elif self.x_dim == 8:
            self.radius = 0.15
        else:
            raise NotImplementedError

        self.cost_spotlight_k = None
        self.cost_p_norm = None
        self.cost_max_noise = 1e-5
        self.cost_discount = 0.0
        self.cost_discount_threshold = 0.0

        if args.cost_fn == "euclidean":
            self.cost_spotlight_k = 1
            self.cost_p_norm = 2
        elif args.cost_fn == "manhattan":
            self.cost_spotlight_k = 1
            self.cost_p_norm = 1
        elif args.cost_fn == "r-spotlight":
            self.cost_spotlight_k = 1e3
            self.cost_p_norm = 2
        elif args.cost_fn == "non-markovian":
            self.cost_spotlight_k = 1
            self.cost_p_norm = 2
            self.cost_discount = 0.1
            self.cost_discount_threshold = 5 * self.radius
        else:
            raise NotImplementedError

        # Random select initial points
        self.bounds = np.array(self.bounds)
        if self.bounds.ndim < 2 or self.bounds.shape[0] < self.x_dim:
            self.bounds = np.tile(self.bounds, [self.x_dim, 1])

        local_bounds = np.zeros_like(self.bounds)
        local_bounds[..., 1] = 1

        n_partitions = int(self.n_initial_points ** (1 / self.x_dim))
        remaining_points = self.n_initial_points - n_partitions**self.x_dim
        ranges = np.linspace(
            local_bounds[..., 0], local_bounds[..., 1], n_partitions + 1
        ).T
        range_bounds = np.stack((ranges[:, :-1], ranges[:, 1:]), axis=-1)
        cartesian_idxs = np.array(
            np.meshgrid(*([list(range(n_partitions))] * self.x_dim))
        ).T.reshape(-1, self.x_dim)
        cartesian_rb = range_bounds[list(range(self.x_dim)), cartesian_idxs]

        self.initial_points = np.concatenate(
            (
                np.random.uniform(
                    low=cartesian_rb[..., 0],
                    high=cartesian_rb[..., 1],
                    size=[n_partitions**self.x_dim, self.x_dim],
                ),
                np.random.uniform(
                    low=local_bounds[..., 0],
                    high=local_bounds[..., 1],
                    size=[remaining_points, self.x_dim],
                ),
            ),
            axis=0,
        )

        if self.env_name == "SynGP":
            self.initial_points[-1] = [0.725, 0.75]
        elif self.env_name == "HolderTable":
            self.initial_points[-1] = [0.5, 0.5]
        elif self.env_name == "EggHolder":
            self.initial_points[-1] = [0.5, 0.5]
        elif self.env_name == "Alpine":
            self.initial_points[-1] = [0.5, 0.5]

        # * np.max(self.bounds[..., 1] - self.bounds[..., 0]) / 100
        self.env_noise = args.env_noise
        self.bounds = torch.tensor(
            self.bounds, dtype=self.torch_dtype, device=self.device
        )
        self.save_dir = f"./results/{args.env_name}_{args.env_noise}{'_discretized' if args.env_discretized else ''}/{args.algo}_{args.cost_fn}_seed{self.seed}"

        if args.env_discretized:
            self.env_discretized = True
            self.num_categories = 20
        else:
            self.env_discretized = False
            self.num_categories = None

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


def run_exp(parms, env) -> None:
    """Run experiment.

    Args:
        parms (Parameter): List of input parameters
        env: Environment
    """
    data_x = torch.tensor(
        parms.initial_points,
        device=parms.device,
        dtype=parms.torch_dtype,
    )
    # >>> n_initial_points x dim

    data_y = env(data_x).reshape(-1, 1)
    # >>> n_initial_points x 1

    data_hidden_state = torch.ones(
        [parms.n_initial_points, parms.hidden_dim],
        device=parms.device,
        dtype=parms.torch_dtype,
    )

    actor = Actor(parms=parms)
    fill_value = float("nan")
    continue_iter = 0
    buffer = TensorDict(
        dict(
            x=torch.full(
                (parms.algo_n_iterations, parms.x_dim),
                fill_value,
                dtype=parms.torch_dtype,
            ),
            y=torch.full(
                (parms.algo_n_iterations, 1),
                fill_value,
                dtype=parms.torch_dtype,
            ),
            h=torch.full(
                (parms.algo_n_iterations, parms.hidden_dim),
                fill_value,
                dtype=parms.torch_dtype,
            ),
            cost=torch.full(
                (parms.algo_n_iterations,),
                fill_value,
                dtype=parms.torch_dtype,
            ),
            runtime=torch.full(
                (parms.algo_n_iterations,),
                fill_value,
                dtype=parms.torch_dtype,
            ),
        ),
        batch_size=[parms.algo_n_iterations],
        device=parms.device,
    )

    if parms.env_discretized:
        embedder = DiscreteEmbbeder(
            num_categories=parms.num_categories,
            bounds=torch.stack(
                [torch.zeros(parms.x_dim), torch.ones(parms.x_dim)], dim=1
            ),
        ).to(device=parms.device, dtype=parms.torch_dtype)
    else:
        embedder = None

    if parms.cont:
        # Load buffers from previous iterations
        buffer_old = torch.load(
            os.path.join(parms.save_dir, "buffer.pt"), map_location=parms.device
        )
        for key in list(buffer_old.keys()):
            buffer[key] = buffer_old[key]
        for idx, x in enumerate(buffer_old["x"]):
            if torch.isnan(x).any():
                continue_iter = idx - 1
                break
        del buffer_old
        torch.cuda.empty_cache()
        print("Continue from iteration: {}".format(continue_iter))

    else:
        buffer["x"][: parms.n_initial_points] = data_x
        buffer["y"][: parms.n_initial_points] = data_y
        buffer["h"][: parms.n_initial_points] = data_hidden_state

    # Set start iteration
    continue_iter = continue_iter if continue_iter != 0 else parms.n_initial_points

    # Warm-up parameters
    surr_model = SingleTaskGP(
        buffer["x"][:continue_iter],
        buffer["y"][:continue_iter],
        # input_transform=Normalize(
        #     d=parms.x_dim, bounds=parms.bounds.T),
        covar_module=parms.kernel,
    ).to(parms.device)
    mll = ExactMarginalLogLikelihood(surr_model.likelihood, surr_model)
    fit_gpytorch_mll(mll)
    actor.construct_acqf(surr_model=surr_model, buffer=buffer[:continue_iter])
    actor.reset_parameters(buffer=buffer[:continue_iter], embedder=embedder)

    # Run BO loop
    for i in range(continue_iter, parms.algo_n_iterations):
        # Initialize model (which is the GP in this case)
        surr_model = SingleTaskGP(
            buffer["x"][:i],
            buffer["y"][:i],
            # input_transform=Normalize(
            #     d=parms.x_dim, bounds=parms.bounds.T),
            covar_module=parms.kernel,
        ).to(parms.device)
        mll = ExactMarginalLogLikelihood(surr_model.likelihood, surr_model)
        fit_gpytorch_mll(mll)

        # Adjust lookahead steps
        if actor.algo_lookahead_steps > 1 and (
            parms.algo_n_iterations - i < actor.algo_lookahead_steps
        ):
            actor.algo_lookahead_steps -= 1
            if not parms.amortized:
                actor.reset_parameters(buffer=buffer[:i], embedder=embedder)

        # Construct acqf
        actor.construct_acqf(surr_model=surr_model, buffer=buffer[:i])

        # Query and observe next point
        query_start_time = time.time()
        output = actor.query(buffer=buffer, iteration=i, embedder=embedder)
        query_end_time = time.time()

        # Save output to buffer
        buffer["x"][i] = output["next_X"]
        buffer["y"][i] = env(output["next_X"])
        if parms.amortized:
            buffer["h"][i] = output["hidden_state"]
        buffer["cost"][i] = output["cost"]
        buffer["runtime"][i] = query_end_time - query_start_time

        # Save buffer to file after each iteration
        torch.save(buffer, f"{parms.save_dir}/buffer.pt")
        print("Buffer saved to file.")

        # Save model to file after each iteration
        torch.save(surr_model.state_dict(), f"{parms.save_dir}/surr_model_{i}.pt")
        print("Model saved to file.")

        # Report to wandb
        logging_data = {
            "x": buffer["x"][i].tolist(),
            "y": buffer["y"][i].tolist(),
            "cost": buffer["cost"][i].item(),
            "runtime": buffer["runtime"][i].item(),
        }
        print(logging_data)
        wandb.log(logging_data)


if __name__ == "__main__":
    # WandB start
    wandb.init(project="nonmyopia")

    # Parse args
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--task", type=str, default="topk")
    parser.add_argument("--env_name", type=str, default="SynGP")
    parser.add_argument("--env_noise", type=float, default=0.0)
    parser.add_argument("--env_discretized", type=str2bool, default=False)
    parser.add_argument("--algo", type=str, default="HES-TS-AM-20")
    parser.add_argument("--cost_fn", type=str, default="r-spotlight")
    parser.add_argument("--plot", type=str2bool, default=False)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--cont", type=str2bool, default=False)
    args = parser.parse_args()

    set_seed(args.seed)

    local_parms = Parameters(args)
    make_save_dir(local_parms)

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

    # Run experiments
    run_exp(local_parms, env)

    # WandB end
    wandb.finish()
