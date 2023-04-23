#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement a BO loop."""

import torch
from argparse import Namespace
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from experiment.checkpoint_manager import pickle_trial_info
from gpytorch.mlls import ExactMarginalLogLikelihood

from models.actor import Actor
from utils.utils import set_seed


def run(parms, env, metrics) -> None:
    """Run experiment.

    Args:
        parms (Parameter): List of input parameters
        env: Environment
        metrics (dict): Dictionary of metrics
    """

    set_seed(parms.seed)

    # Generate initial observations and initialize model
    bd_l, bd_u = parms.bounds
    data_x = torch.rand(
        [parms.n_initial_points, parms.x_dim], device=parms.device, dtype=parms.torch_dtype
    )
    data_x = data_x * (bd_l - bd_u) + bd_l
    # >>> n x dim

    data_y = env(data_x)  # n x 1
    if parms.func_is_noisy:
        data_y = data_y + parms.func_noise * torch.randn_like(data_y, parms.torch_dtype)

    buffer = Namespace(x=data_x, y=data_y.reshape(-1, 1))

    WM = SingleTaskGP(
        buffer.x,
        buffer.y,
        outcome_transform=Standardize(1),
    ).to(parms.device)
    mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
    fit_gpytorch_model(mll)

    actor = Actor(parms=parms, WM=WM, buffer=buffer)

    metrics[f"eval_metric_{parms.algo}_{parms.seed}"] = [
        float("nan")
    ] * parms.n_iterations
    metrics[f"optimal_action_{parms.algo}_{parms.seed}"] = [
        float("nan")
    ] * parms.n_iterations

    lookahead_steps = parms.lookahead_steps

    # Run BO loop
    for iteration in range(parms.n_iterations):
        next_x, optimal_actions, eval_metric = actor.query(buffer, iteration)
        next_y = env(next_x).reshape(-1, 1)

        # Update training points
        buffer.x = torch.cat([buffer.x, next_x])
        buffer.y = torch.cat([buffer.y, next_y])

        # Evaluate
        eval_metric = eval_metric.cpu().squeeze()
        optimal_actions = optimal_actions.cpu().squeeze()
        metrics[f"eval_metric_{parms.algo}_{parms.seed}"][
            iteration
        ] = eval_metric.item()
        metrics[f"optimal_action_{parms.algo}_{parms.seed}"][
            iteration
        ] = optimal_actions.numpy().tolist()
        print(f"Eval metric: {eval_metric.item()}")

        # Pickle trial info at each iteration
        pickle_trial_info(
            parms,
            buffer,
            metrics[f"eval_metric_{parms.algo}_{parms.seed}"],
            metrics[f"optimal_action_{parms.algo}_{parms.seed}"],
        )

        # Draw posterior
        parms.eval_function(
            config=parms,
            env=env,
            acqf=actor.acqf,
            buffer=buffer,
            iteration=iteration,
            optimal_actions=optimal_actions,
        )

        # Fit the model
        WM = SingleTaskGP(
            buffer.x,
            buffer.y,
            outcome_transform=Standardize(1),
        )
        mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
        fit_gpytorch_model(mll)
        WM = WM.to(parms.device)

        if not parms.learn_hypers:
            WM = env.set_ground_truth_GP_hyperparameters(WM)

        # Set WM to actor
        actor.acqf.model = WM

        # Adjust lookahead steps
        if lookahead_steps > 1 and iteration >= parms.lookahead_warmup:
            lookahead_steps -= 1
            actor.lookahead_steps = lookahead_steps
            if parms.algo in ["HES", "qMSL"]:
                model = actor.acqf.model
                actor.acqf = actor.construct_acqf(WM=model, buffer=buffer)

            if not parms.use_amortized_optimization:
                actor.maps = []

    # Optional final evaluation
    if parms.final_eval_function is not None:
        with torch.no_grad():
            parms.final_eval_function(metrics["eval_metric_list"], parms)
