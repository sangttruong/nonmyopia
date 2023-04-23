#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Implement Multi-step Expected H-Information Gain acquisition function.
"""

import copy
import torch
import torch.nn as nn
from torch import Tensor
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from typing import List, Dict, Optional, Type, Tuple


class qMultiStepEHIG(MCAcquisitionFunction):
    """qMultiStepEHIG Class."""

    def __init__(
        self,
        model,
        loss_function_class: Type[nn.Module],
        loss_function_hyperparameters: Dict[str, int],
        cost_function_class: Type[nn.Module],
        cost_function_hyperparameters: Dict[str, int],
        lookahead_steps: int,
        n_actions: int,
        n_fantasy_at_design_pts: Optional[List[int]] = 64,
        n_fantasy_at_action_pts: Optional[int] = 64,
        design_sampler: Optional[MCSampler] = None,
        action_sampler: Optional[MCSampler] = None,
        maps: Optional[List[nn.Module]] = None,
    ) -> None:
        """Batch H-Entropy Search using multi-step optimization.

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
            design_sampler (Optional[MCSampler], optional): The samplers
                used to sample fantasized outcomes at each design point.
                Optional if `n_fantasy_at_design_pts` is specified.
                Defaults to None.
            action_sampler (Optional[MCSampler], optional): The sampler
                used to sample fantasized outcomes at each action point.
                Optional if `n_fantasy_at_design_pts` is specified.
                Defaults to None.
            maps (Optional[List[nn.Module]], optional): List of parameters
                for optimizing. Defaults to None.
        """
        super().__init__(model=model)
        self.model = model
        self.lookahead_steps = lookahead_steps
        self.n_actions = n_actions
        self.maps = maps
        self.cost_function = cost_function_class(**cost_function_hyperparameters)
        self.loss_function = loss_function_class(**loss_function_hyperparameters)
        self.design_samplers = []
        self.n_fantasy_at_design_pts = []
        for i in range(lookahead_steps):
            sampler, n_fantasy = set_sampler_and_n_fantasy(
                sampler=design_sampler, n_fantasy=n_fantasy_at_design_pts[i]
            )
            self.design_samplers.append(sampler)
            self.n_fantasy_at_design_pts.append(n_fantasy)

        action_sampler, n_fantasy_at_action_pts = set_sampler_and_n_fantasy(
            sampler=action_sampler, n_fantasy=n_fantasy_at_action_pts
        )
        self.action_sampler = action_sampler
        self.n_fantasy_at_action_pts = n_fantasy_at_action_pts

        self.use_amortized_map = True if isinstance(maps, nn.Module) else False

    def forward(
        self,
        prev_X: Tensor,
        prev_y: Tensor,
        prev_hid_state: Tensor,
        previous_cost: float,
    ) -> Dict[str, Tensor]:
        """
        Evaluate qMultiStepEHIG objective (q-MultistepHES).

        Args:
            prev_X (Tensor): A tensor of shape `batch x x_dim`.
            prev_y (Tensor): A tensor of shape `batch x y_dim`.
            prev_hid_state (Tensor): A tensor of shape `batch x hidden_dim`.
            previous_cost (double): For cummulative cost function

        Returns:
            dict: A dictionary contains acqf_values, first_X, 
                first_hidden_state, actions and X.
        """
        n_restarts = prev_X.shape[0]
        x_dim = prev_X.shape[1]
        y_dim = prev_y.shape[1]
        previous_Xy = torch.cat((prev_X, prev_y), dim=-1)

        fantasized_model = copy.deepcopy(self.model)
        X_returned = []
        hidden_state_returned = []

        for step in range(self.lookahead_steps):
            # condition on X[step], then sample, then condition on (x, y)
            n_fantasies = self.n_fantasy_at_design_pts[step]

            # Pass through RNN
            if self.use_amortized_map:
                X, hidden_state = self.maps(
                    x=previous_Xy, 
                    prev_hid_state=prev_hid_state, 
                    return_actions=False
                )
            else:
                return self.maps[step], prev_hid_state

            new_shape = [n_fantasies] * step + [n_restarts, x_dim]
            X = X.reshape(*new_shape)
            # >>> num_x_{step} * x_dim

            X_returned.append(X)
            hidden_state_returned.append(hidden_state)

            # Sample posterior
            ppd = fantasized_model.posterior(X)
            ys = self.design_samplers[step](ppd)
            # >>> n_samples * num_x_{step} * y_dim

            X_expanded_shape = [ys.shape[0]] + [-1] * len(new_shape)
            X_expanded = X[None, ...].expand(*X_expanded_shape)
            Xy = torch.cat((X_expanded, ys), dim=-1)
            # >>> n_samples * num_x_{step} * 1 * dim

            # Update previous_Xy
            previous_Xy = Xy.reshape(-1, x_dim + y_dim)
            # >>> (n_samples * num_x_{step}) * seq_length * y_dim

            # Update conditions
            fantasized_model = fantasized_model.condition_on_observations(X, ys)

            # Update hidden state
            prev_hid_state = hidden_state[None, ...].expand(n_fantasies, -1, -1)

            prev_hid_state = prev_hid_state.reshape(
                -1, hidden_state.shape[-1]
            )

        # Compute actions
        if self.use_amortized_map:
            actions, hidden_state = self.maps(
                x=previous_Xy,
                prev_hid_state=prev_hid_state, 
                return_actions=True
            )
        else:
            return self.maps[self.lookahead_steps], hidden_state

        new_shape = self.n_fantasy_at_design_pts + [
            n_restarts, self.n_actions, x_dim,
        ]
        actions = actions.reshape(*new_shape)

        post_pred_dist = [
            self.model.posterior(actions[..., k, :]) for k in range(self.n_actions)
        ]

        action_yis = [self.action_sampler(ppd) for ppd in post_pred_dist]
        action_yis = torch.stack(action_yis, dim=-2).squeeze(-1)
        # >> Tensor[*[n_samples]*i, n_restarts, 1, 1]

        acqf_values = self.loss_function(actions, action_yis)

        # TODO: add cost to acqf_values
        # Calculate cost
        # total_cost = previous_cost
        # for ... 
        #     total_cost = total_cost + self.cost_function(
        #         previous_Xy[:, : x_dim], X
        #     )
        #     total_cost = (
        #         total_cost[None, ...].expand(n_fantasies, -1).reshape(-1)
        #     )

        # new_shape = [-1, self.n_actions, x_dim]
        # actions = actions.reshape(*new_shape)
        # for ai in range(self.n_actions):
        #     total_cost = total_cost + self.cost_function(
        #         previous_Xy[:, :x_dim], actions[..., ai, :]
        #     )

        # new_shape = self.n_fantasy_at_design_pts + [
        #     n_restarts
        # ]
        # total_cost = total_cost.reshape(*new_shape)
        # acqf_values = acqf_values + total_cost
        
        while len(acqf_values.shape) > 1:
            acqf_values = acqf_values.mean(dim=0)
        # >>> batch number of x_0

        return {
            "acqf_values": acqf_values,
            "X": X_returned,
            "actions": actions,
            "hidden_state": hidden_state_returned,
        }


def set_sampler_and_n_fantasy(
    sampler: Optional[MCSampler], n_fantasy: Optional[int]
) -> Tuple[MCSampler, int]:
    r"""Create samplers and sample posteriror predictives.

    Args:
        sampler: The sampler to use. If None, a SobolQMCNormalSampler will be
            created with shape of `n_fantasy`.
        n_fantasy: The number of fantasy samples of the sampler. If None, the
            sampler sample shape will be used.

    Returns:
        A tuple of the sampler and the number of fantasy samples.
    """
    if sampler is None:
        if n_fantasy is None:
            raise ValueError("Must specify `n_fantasy` if no `sampler` is provided.")
        # base samples should be fixed for joint optimization
        sampler = SobolQMCNormalSampler(
            sample_shape=n_fantasy,
            resample=False,
            collapse_batch_dims=True,
        )
    elif n_fantasy is not None:
        if sampler.sample_shape != torch.Size([n_fantasy]):
            raise ValueError("The sampler shape must match {n_fantasy}.")
    else:
        n_fantasy = sampler.sample_shape[0]

    return sampler, n_fantasy


class qLossFunctionTopK(nn.Module):
    """Loss function for Top-K task."""

    def __init__(self, dist_weight=1.0, dist_threshold=0.5) -> None:
        r"""Batch loss function for the task of finding top-K.

        Args:
            loss_function_hyperparameters: hyperparameters for the loss function class.
        """
        super().__init__()
        self.register_buffer("dist_weight", torch.as_tensor(dist_weight))
        self.register_buffer("dist_threshold", torch.as_tensor(dist_threshold))

    def forward(self, A: Tensor, Y: Tensor) -> Tensor:
        r"""Evaluate batch loss function on a tensor of actions.

        Args:
            A: Actor tensor with shape `batch_size x n_fantasy_at_design_pts
                x num_actions x action_dim`.
            Y: Fantasized sample with shape `n_fantasy_at_action_pts x
                n_fantasy_at_design_pts x batch_size x num_actions`.

        Returns:
            A Tensor of shape `n_fantasy_at_action_pts x batch`.
        """
        Y = Y.sum(dim=-1).mean(dim=0)
        # >>> n_fantasy_at_design_pts x batch_size

        num_actions = A.shape[-2]

        dist_reward = 0
        if num_actions >= 2:
            A_distance = torch.cdist(A.contiguous(), A.contiguous(), p=1.0)
            A_distance_triu = torch.triu(A_distance)
            # >>> n_fantasy_at_design_pts x batch_size x num_actions x num_actions

            A_distance_triu[A_distance_triu > self.dist_threshold] = self.dist_threshold
            denominator = num_actions * (num_actions - 1) / 2.0
            dist_reward = A_distance_triu.sum((-1, -2)) / denominator
            # >>> n_fantasy_at_design_pts x batch_size

        q_hes = Y + self.dist_weight * dist_reward
        # >>> n_fantasy_at_design_pts x batch_size

        return q_hes


class qCostFunctionSpotlight(nn.Module):
    def __init__(self, radius: float) -> None:
        super().__init__()
        self.register_buffer("radius", torch.as_tensor(radius))
    
    def forward(self, prev_X, current_X):
        diff = torch.sqrt(torch.pow(current_X - prev_X, 2).sum(-1))
        nb_idx = diff < self.radius
        diff = diff * (1 - nb_idx.float()) * 100
        return diff


class qCostFunctionL2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, prev_X, current_X):
        diff = torch.sqrt(torch.pow(current_X - prev_X, 2).sum(-1))
        return diff
