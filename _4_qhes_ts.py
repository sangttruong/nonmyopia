#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Multi-step H-Entropy Search with one-shot optimization."""

from typing import Dict, List, Optional, Tuple, Type

import copy
import torch
import torch.nn as nn
from torch import Tensor
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.sampling.base import MCSampler
from botorch import settings
from botorch.models.utils.assorted import fantasize as fantasize_flag
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.gp_sampling import GPDraw
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths
from _4_qhes import set_sampler_and_n_fantasy
from _6_samplers import PosteriorMeanSampler


class qMultiStepHEntropySearchTS(MCAcquisitionFunction):
    """qMultiStep H-Entropy Search Class."""

    def __init__(
        self,
        model,
        loss_function_class: Type[nn.Module],
        loss_func_hypers: Dict[str, int],
        cost_function_class: Type[nn.Module],
        cost_func_hypers: Dict[str, int],
        lookahead_steps: int,
        n_actions: int,
        n_fantasy_at_design_pts: Optional[List[int]] = 64,
        n_fantasy_at_action_pts: Optional[int] = 64,
        design_samplers: Optional[MCSampler] = None,
        action_sampler: Optional[MCSampler] = None,
    ) -> None:
        """Batch multip-step H-Entropy Search using one-shot optimization.

        Args:
            model: A fitted model. Must support fantasizing.
            loss_function_class (Type[nn.Module]): The loss function class
                that is used to compute the expected loss
                of the fantasized actions.
            cost_function class (Optional[nn.Module]): Cost function
                class that is used to compute the cost of the fantasized
                trajectories.
            lookahead_steps (int): Number of lookahead steps
            n_actions (int): Number of actions
            n_fantasy_at_design_pts (Optional[List[int]], optional): Number
                of fantasized outcomes for each design point. Must match
                the sample shape of `design_sampler` if specified.
                Defaults to 64.
            n_fantasy_at_action_pts (Optional[int], optional): Number of
                fantasized outcomes for each action point. Must match the
                sample shape of `action_sampler` if specified.
                Defaults to 64.
            design_samplers (Optional[MCSampler], optional): The samplers
                used to sample fantasized outcomes at each design point.
                Optional if `n_fantasy_at_design_pts` is specified.
                Defaults to None.
            action_sampler (Optional[MCSampler], optional): The sampler
                used to sample fantasized outcomes at each action point.
                Optional if `n_fantasy_at_design_pts` is specified.
                Defaults to None.
        """
        super().__init__(model=model)
        self.model = model
        self._model = None
        self.algo_lookahead_steps = lookahead_steps
        self.n_actions = n_actions
        self.cost_function = cost_function_class(**cost_func_hypers)
        self.loss_function = loss_function_class(**loss_func_hypers)
        self.design_samplers = []
        self.n_fantasy_at_design_pts = []
        for i in range(lookahead_steps):
            if design_samplers is not None:
                sampler = design_samplers[i]
            else:
                sampler = None

            sampler, n_fantasy = set_sampler_and_n_fantasy(
                sampler=sampler, n_fantasy=n_fantasy_at_design_pts[i]
            )
            self.design_samplers.append(sampler)
            self.n_fantasy_at_design_pts.append(n_fantasy)

        action_sampler, n_fantasy_at_action_pts = set_sampler_and_n_fantasy(
            sampler=action_sampler, n_fantasy=n_fantasy_at_action_pts
        )
        self.action_sampler = action_sampler
        self.n_fantasy_at_action_pts = n_fantasy_at_action_pts

    def dump_model(self):
        """Dump model."""
        self._model = copy.deepcopy(self.model)

    def clean_dump_model(self):
        """Clean dump model."""
        del self._model
        torch.cuda.empty_cache()
        self._model = None

    def forward(
        self,
        prev_X: Tensor,
        prev_y: Tensor,
        prev_hid_state: Tensor,
        maps: List[nn.Module],
        embedder: nn.Module = None,
        prev_cost: float = 0.0
    ) -> Dict[str, Tensor]:
        """
        Evaluate qMultiStepEHIG objective (q-MultistepHES).

        Args:
            prev_X (Tensor): A tensor of shape `batch x x_dim`.
            prev_y (Tensor): A tensor of shape `batch x y_dim`.
            prev_hid_state (Tensor): A tensor of shape `batch x hidden_dim`.
            maps (Optional[List[nn.Module]], optional): List of parameters
                for optimizing. Defaults to None.

        Returns:
            dict: A dictionary contains acqf_loss, X, actions and hidden_state.
        """
        use_amortized_map = True if isinstance(maps, nn.Module) else False
        n_restarts = prev_X.shape[0]
        x_dim = prev_X.shape[1]
        y_dim = prev_y.shape[1]
        num_categories = prev_X.shape[2] if embedder is not None else 0
        previous_X = prev_X
        previous_y = prev_y
        previous_cost = prev_cost

        X_returned = []
        hidden_state_returned = []
        for step in range(self.algo_lookahead_steps):
            # Draw new f ~ p(f|D)
            # self.f = GPDraw(self.model, seed=0)
            self.f = draw_matheron_paths(self.model, torch.Size([1]))

            # condition on X[step], then sample, then condition on (x,prev_X y)
            if use_amortized_map:
                X, hidden_state = maps(
                    x=previous_X,
                    y=previous_y,
                    prev_hid_state=prev_hid_state,
                    return_actions=False,
                )
                # >>> n_restart x x_dim x (num_categories)
            else:
                X, hidden_state = maps[step], prev_hid_state

            n_fantasies = self.n_fantasy_at_design_pts[step]
            if num_categories > 0:
                X_shape = self.n_fantasy_at_design_pts[:step][::-1] + [
                    n_restarts,
                    1,
                    x_dim,
                    num_categories,
                ]
            else:
                X_shape = self.n_fantasy_at_design_pts[:step][::-1] + [
                    n_restarts,
                    1,
                    x_dim,
                ]

            X = X.reshape(*X_shape)
            # >>> num_x_{step} x 1 x x_dim x (num_categories)

            X_expanded_shape = [n_fantasies] + [-1] * len(X_shape)
            X_expanded = X[None, ...].expand(*X_expanded_shape)
            # >>> n_samples x num_x_{step} x 1 x dim x (num_categories)

            if embedder is not None:
                X = embedder.encode(X)
                # >>> num_x_{step} * x_dim

            X_returned.append(X)
            hidden_state_returned.append(hidden_state)

            # Sample posterior
            ys = self.f(X.squeeze(dim=list(range(X.dim() - 3)))).unsqueeze(0)
            
            # Update previous_Xy
            if num_categories > 0:
                previous_X = X_expanded.reshape(-1, x_dim, num_categories)
            else:
                previous_X = X_expanded.reshape(-1, x_dim)
            previous_y = ys.reshape(-1, y_dim)
            # >>> (n_samples * num_x_{step}) * seq_length * y_dim

            # Update hidden state
            prev_hid_state = hidden_state[None, ...]
            prev_hid_state = prev_hid_state.expand(n_fantasies, -1, -1)
            prev_hid_state = prev_hid_state.reshape(-1, hidden_state.shape[-1])

        # Compute actions
        if use_amortized_map:
            actions, hidden_state = maps(
                x=previous_X,
                y=previous_y,
                prev_hid_state=prev_hid_state,
                return_actions=True,
            )
        else:
            actions = maps[self.algo_lookahead_steps]

        if embedder is not None:
            actions = embedder.encode(actions)
        action_shape = self.n_fantasy_at_design_pts[::-1] + [
            n_restarts,
            self.n_actions,
            x_dim,
        ]
        actions = actions.reshape(*action_shape)
        action_yis = self.f(
            actions.squeeze(dim=list(range(actions.dim() - 3)))
        ).unsqueeze(0)
        # >> Tensor[*[n_samples]*i, n_restarts, n_actions]

        # Calculate loss value
        acqf_loss = self.loss_function(actions, action_yis)

        # Calculate cost value
        first_prev_X = prev_X[:, None, ...]
        if embedder is not None:
            first_prev_X = embedder.encode(first_prev_X)
        acqf_cost = self.cost_function(
            prev_X=first_prev_X, current_X=X_returned[0], previous_cost=previous_cost
        )
        for i in range(self.algo_lookahead_steps - 1):
            cX = X_returned[i + 1]
            pX = X_returned[i][None, ...].expand_as(cX)
            acqf_cost = acqf_cost + self.cost_function(
                prev_X=pX, current_X=cX, previous_cost=acqf_cost + previous_cost
            )
        for i in range(self.n_actions):
            cX = actions[..., i: i + 1, :]
            pX = X_returned[-1][None, ...].expand_as(cX)
            acqf_cost = acqf_cost + self.cost_function(
                prev_X=pX, current_X=cX, previous_cost=acqf_cost + previous_cost
            )
        acqf_cost = acqf_cost.squeeze(dim=-1).sum(dim=-1)
        
        # Reduce dimensions
        while len(acqf_loss.shape) > 1:
            acqf_loss = acqf_loss.mean(dim=0)
        while len(acqf_cost.shape) > 1:
            acqf_cost = acqf_cost.mean(dim=0)
        # >>> batch number of x_0

        return {
            "acqf_loss": acqf_loss,
            "acqf_cost": acqf_cost,
            "X": X_returned,
            "actions": actions,
            "hidden_state": hidden_state_returned,
        }
