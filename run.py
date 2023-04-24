#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement a BO loop."""

import torch
import random
import numpy as np
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from models.actor import Actor
from utils.utils import set_seed
from utils.plot import eval_and_plot_2D
from tensordict import TensorDict
from amortized_network import Project2Range


def run(parms, env, metrics) -> None:
    """Run experiment.

    Args:
        parms (Parameter): List of input parameters
        env: Environment
        metrics (dict): Dictionary of metrics
    """

    random.seed(parms.seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(parms.seed)
    torch.manual_seed(parms.seed)
    torch.cuda.manual_seed_all(parms.seed)
    torch.cuda.manual_seed_all(parms.seed)

    # Generate initial observations and initialize model
    project2range = Project2Range(*parms.bounds)
    data_x = torch.rand(
        [parms.n_initial_points, parms.x_dim],
        device=parms.device,
        dtype=parms.torch_dtype,
    )
    data_x = project2range(data_x)
    # >>> n_initial_points x dim

    data_y = env(data_x).reshape(-1, 1)
    # >>> n_initial_points x 1
    if parms.func_is_noisy:
        data_y = data_y + parms.func_noise * torch.randn_like(data_y, parms.torch_dtype)

    buffer_size = parms.n_initial_points + parms.n_iterations
    fill_value = float("nan")
    buffer = TensorDict(
        dict(
            x=torch.full((buffer_size, parms.x_dim), fill_value),
            y=torch.full((buffer_size, 1), fill_value),
            h=torch.full((buffer_size, parms.hidden_dim), fill_value),
        ),
        batch_size=[buffer_size],
        device=parms.device,
        dtype=parms.torch_dtype,
    )

    buffer["x"][: parms.n_initial_points] = data_x
    buffer["y"][: parms.n_initial_points] = data_y

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
        next_x = actor.query(buffer, i)
        next_y = env(next_x).reshape(-1, 1)
        # Evaluate and plot
        eval_and_plot_2D(
            func=env, 
            cfg=parms, 
            qhes=actor.acqf, 
            next_x=next_x, 
            data=buffer,
            iteration=i
        )

        buffer["x"][i] = next_x
        buffer["y"][i] = next_y
