#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement a BO loop."""

import torch
import random
import numpy as np

from tensordict import TensorDict
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from _2_actor_dc import Actor
from _5_evalplot import eval_and_plot_1D, eval_and_plot_2D, eval_func
from _9_semifuncs import generate_random_X


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
        eval_and_plot = eval_func

    # Generate initial observations and initialize model
    # data_x = torch.rand(
    #     [parms.n_initial_points, parms.x_dim],
    #     device=parms.device,
    #     dtype=parms.torch_dtype,
    # )

    # Min max scaling
    # data_x = data_x * (parms.bounds[1] - parms.bounds[0]) + parms.bounds[0]
    # >>> n_initial_points x dim

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
    data_y = data_y + parms.func_noise * torch.randn_like(
        data_y, dtype=parms.torch_dtype
    )

    data_hidden_state = torch.ones(
        [parms.n_initial_points, parms.hidden_dim],
        device=parms.device,
        dtype=parms.torch_dtype,
    )

    buffer_size = parms.n_iterations
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

    actor = Actor(parms=parms)
    
    embedder = DiscreteEmbbeder(    
        num_categories=parms.num_categories,
        bounds=parms.bounds,
    ).to(device=parms.device, dtype=parms.torch_dtype)
    
    WM = SingleTaskGP(
            buffer["x"][:parms.n_initial_points],
            buffer["y"][:parms.n_initial_points],
            outcome_transform=Standardize(1),
            covar_module=parms.kernel,
        ).to(parms.device)
        
    mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
    fit_gpytorch_model(mll)
    actor.construct_acqf(WM=WM, buffer=buffer[:parms.n_initial_points])
    
    real_loss, _ = eval_and_plot(
            func=env,
            wm=WM,
            cfg=parms,
            acqf=actor.acqf,
            buffer=buffer[:parms.n_initial_points-1],
            next_x=buffer['x'][parms.n_initial_points-1],
            actions=None,
            iteration=parms.n_initial_points-1,
            n_space=parms.num_categories,
            embedder=embedder
        )
    return
    real_loss, _ = eval_func(
        func=env,
        cfg=parms,
        acqf=actor.acqf
    )
    buffer["real_loss"][parms.n_initial_points-1] = real_loss
    
    # Run BO loop
    for i in range(parms.n_initial_points, parms.n_iterations):
        # Initialize model (which is the GP in this case)
        WM = SingleTaskGP(
            buffer["x"][:i],
            buffer["y"][:i],
            outcome_transform=Standardize(1),
            covar_module=parms.kernel,
        ).to(parms.device)
        
        mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
        fit_gpytorch_model(mll)
        # Adjust lookahead steps
        if actor.lookahead_steps > 1 and (parms.n_iterations - i < actor.lookahead_steps):
            actor.lookahead_steps -= 1
        actor.construct_acqf(WM=WM, buffer=buffer[:i])

        # Query and observe next point
        next_x, hidden_state, actions = actor.query(
            buffer=buffer, 
            iteration=i,
            embedder=embedder
        )
        
        next_y = env(next_x).reshape(-1, 1)

        # Evaluate and plot
        real_loss, _ = eval_and_plot(
            func=env,
            wm=WM,
            cfg=parms,
            acqf=actor.acqf,
            buffer=buffer,
            next_x=next_x,
            actions=actions,
            iteration=i,
            n_space=parms.num_categories,
            embedder=embedder
        )

        buffer["x"][i] = next_x
        buffer["y"][i] = next_y
        buffer["h"][i] = hidden_state
        buffer["real_loss"][i] = real_loss

        print("Real loss:", real_loss.item())
        
        # Save buffer to file after each iteration
        torch.save(buffer, f'{parms.save_dir}/buffer.pt')
        print("Buffer saved to file.")
            
        # Save model to file after each iteration
        torch.save(WM.state_dict(), f'{parms.save_dir}/world_model.pt')
        print("Model saved to file.")

    return buffer["real_loss"].cpu().detach().numpy().tolist()


class DiscreteEmbbeder:
    def __init__(self, num_categories, bounds):
        self.num_categories = num_categories
        self.bounds = bounds
        self.range_size = (bounds[1] - bounds[0]) / num_categories
        midpoints = []
        for i in range(self.num_categories):
            midpoint = self.bounds[0] + self.range_size * (i + 0.5)
            # rand uniform in range (- range_size/2, + range_size/2)
            rand = torch.rand(1).item() * self.range_size - self.range_size / 2
            midpoint = midpoint + rand

            # clip to bounds
            midpoint = max(midpoint, self.bounds[0])
            midpoint = min(midpoint, self.bounds[1])

            midpoints.append(midpoint)
        self.cat_range = torch.tensor(midpoints)

    def encode(self, sentence, *args, **kwargs):
        """ Cat2Con
        Args:
            sentence (torch.Tensor): A tensor of shape `... x num_categories` one
                hot vector
        """
        assert torch.any(sentence >= 0)
        assert torch.any(sentence < self.num_categories)
        return (self.cat_range * sentence).sum(dim=-1)
    
    def decode(self, sentence, *args, **kwargs):
        # Con2Cat
        assert torch.any(sentence >= self.bounds[0])
        assert torch.any(sentence <= self.bounds[1])
        cat = (sentence - self.bounds[0]) / self.range_size
        cat = cat.long()
        cat[cat == self.num_categories] = self.num_categories - 1
        return cat

    def to(self, device="cpu", dtype=torch.float32):
        self.cat_range = self.cat_range.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        return self