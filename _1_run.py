#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement a BO loop."""

import random

import numpy as np
import torch
from _3_amortized_network import Project2Range
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from _2_actor import Actor
from tensordict import TensorDict
from _5_evalplot import eval_and_plot_1D, eval_and_plot_2D


def run(parms, env) -> None:
    """Run experiment.

    Args:
        parms (Parameter): List of input parameters
        env: Environment
    """

    random.seed(parms.seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(parms.seed)
    torch.manual_seed(parms.seed)
    torch.cuda.manual_seed_all(parms.seed)
    torch.cuda.manual_seed_all(parms.seed)

    if parms.x_dim == 1:
        eval_and_plot = eval_and_plot_1D
    elif parms.x_dim == 2:
        eval_and_plot = eval_and_plot_2D
    else:
        print("Plotting is only done when x_dim is 1 or 2.")

    # Generate initial observations and initialize model
    data_x = torch.rand(
        [parms.n_initial_points, parms.x_dim],
        device=parms.device,
        dtype=parms.torch_dtype,
    )
    # Min max scaling
    data_x = data_x * (parms.bounds[1] - parms.bounds[0]) + parms.bounds[0]
    # >>> n_initial_points x dim

    data_y = env(data_x).reshape(-1, 1)
    # >>> n_initial_points x 1
    data_y = data_y + parms.func_noise * torch.randn_like(
        data_y, dtype=parms.torch_dtype
    )

    data_hidden_state = torch.ones(
        [parms.n_initial_points, parms.hidden_dim],
        device=parms.device,
        dtype=parms.torch_dtype,
    )

    buffer_size = parms.n_initial_points + parms.n_iterations
    fill_value = float("nan")
    buffer = TensorDict(
        dict(
            x=torch.full(
                (buffer_size, parms.x_dim),
                fill_value,
                dtype=parms.torch_dtype,
            ),
            y=torch.full(
                (buffer_size, 1),
                fill_value,
                dtype=parms.torch_dtype,
            ),
            h=torch.full(
                (buffer_size, parms.hidden_dim),
                fill_value,
                dtype=parms.torch_dtype,
            ),
            real_loss=torch.full(
                (buffer_size,),
                fill_value,
                dtype=parms.torch_dtype,
            ),
        ),
        batch_size=[buffer_size],
        device=parms.device,
    )

    buffer["x"][: parms.n_initial_points] = data_x
    buffer["y"][: parms.n_initial_points] = data_y
    buffer["h"][: parms.n_initial_points] = data_hidden_state

    lookahead_steps = parms.lookahead_steps
    actor = Actor(parms=parms)

    # Run BO loop
    for i in range(parms.n_initial_points, parms.n_iterations):
        # Initialize model (which is the GP in this case)
        WM = SingleTaskGP(
            buffer["x"][:i],
            buffer["y"][:i],
            outcome_transform=Standardize(1),
        ).to(parms.device)
        mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
        fit_gpytorch_model(mll)
        # Adjust lookahead steps
        j = parms.n_iterations - parms.lookahead_steps
        if lookahead_steps > 1 and i >= j:
            actor.lookahead_steps = lookahead_steps - 1
        actor.construct_acqf(WM=WM, buffer=buffer)

        # Query and observe next point
        next_x, hidden_state = actor.query(buffer, i)
        next_y = env(next_x).reshape(-1, 1)

        # Evaluate and plot
        if parms.x_dim in [1, 2]:
            real_loss = eval_and_plot(
                func=env,
                cfg=parms,
                acqf=actor.acqf,
                buffer=buffer,
                next_x=next_x,
                iteration=i,
            )
        else:
            raise NotImplementedError

        buffer["x"][i] = next_x
        buffer["y"][i] = next_y
        buffer["h"][i] = hidden_state
        buffer["real_loss"][i] = real_loss

    return buffer["real_loss"].cpu().detach().numpy().tolist()