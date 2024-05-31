#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement a BO loop."""

import torch
import random
import numpy as np
import os
import time
import wandb

from tqdm import tqdm
from tensordict import TensorDict
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from _2_actor import Actor
from _5_evalplot import eval_and_plot
from _9_semifuncs import generate_random_X
from _13_embedder import DiscreteEmbbeder


def run(parms, env) -> None:
    """Run experiment.

    Args:
        parms (Parameter): List of input parameters
        env: Environment
    """
    if parms.env_name == "AntBO":
        data_x = generate_random_X(parms.n_initial_points, parms.x_dim)
        data_x = data_x.to(
            device=parms.device,
            dtype=parms.torch_dtype,
        )
    else:
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
            bounds=parms.bounds,
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
    WM = SingleTaskGP(
        buffer["x"][: continue_iter],
        buffer["y"][: continue_iter],
        # input_transform=Normalize(
        #     d=parms.x_dim, bounds=parms.bounds.T),
        # outcome_transform=Standardize(1),
        covar_module=parms.kernel,
    ).to(parms.device)
    mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
    fit_gpytorch_model(mll)
    actor.construct_acqf(WM=WM, buffer=buffer[:continue_iter])
    actor.reset_parameters(buffer=buffer[:continue_iter], embedder=embedder)
    
    # Run BO loop
    for i in range(continue_iter, parms.algo_n_iterations):
        # Initialize model (which is the GP in this case)
        WM = SingleTaskGP(
            buffer["x"][:i],
            buffer["y"][:i],
            # input_transform=Normalize(
            #     d=parms.x_dim, bounds=parms.bounds.T),
            # outcome_transform=Standardize(1),
            covar_module=parms.kernel,
        ).to(parms.device)
        mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
        fit_gpytorch_model(mll)

        # Adjust lookahead steps
        if actor.algo_lookahead_steps > 1 and (
            parms.algo_n_iterations - i < actor.algo_lookahead_steps
        ):
            actor.algo_lookahead_steps -= 1

        # Construct acqf
        actor.construct_acqf(WM=WM, buffer=buffer[:i])

        # Query and observe next point
        query_start_time = time.time()
        output = actor.query(
            buffer=buffer, 
            iteration=i, 
            embedder=embedder
        )
        query_end_time = time.time()

        # Save output to buffer
        buffer["x"][i] = output["next_X"]
        buffer["y"][i] = env(output["next_X"])
        if parms.amortized:
            buffer["h"][i] = output["hidden_state"]
        buffer["cost"][i] = output["cost"]
        buffer["runtime"][i] = query_end_time - query_start_time

        # Evaluate and plot
        ## Separated ##

        # Save buffer to file after each iteration
        torch.save(buffer, f"{parms.save_dir}/buffer.pt")
        print("Buffer saved to file.")

        # Save model to file after each iteration
        torch.save(WM.state_dict(), f"{parms.save_dir}/world_model_{i}.pt")
        print("Model saved to file.")

        # Report to wandb
        wandb.log({
            "x": buffer["x"][i], 
            "y": buffer["y"][i],
            "cost": buffer["cost"][i],
            "runtime": buffer["runtime"][i]
        })
