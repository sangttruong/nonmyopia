#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Multi-step H-Entropy Search with one-shot optimization."""

from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from copy import deepcopy
from torch import Tensor
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.sampling.base import MCSampler

from botorch.sampling.normal import SobolQMCNormalSampler
from _6_samplers import PosteriorMeanSampler


class qMultiStepHEntropySearch(MCAcquisitionFunction):
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
        self.lookahead_steps = lookahead_steps
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

    def forward(
        self,
        prev_X: Tensor,
        prev_y: Tensor,
        prev_hid_state: Tensor,
        maps: List[nn.Module],
        embedder,
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
        num_categories = prev_X.shape[2]
        y_dim = prev_y.shape[1]
        previous_X = prev_X
        previous_y = prev_y

        fantasized_model = deepcopy(self.model)
        X_returned = []
        hidden_state_returned = []
        
        for step in range(self.lookahead_steps):
            # condition on X[step], then sample, then condition on (x, y)
            if use_amortized_map:
                X, hidden_state = maps(
                    x=previous_X, y=previous_y, prev_hid_state=prev_hid_state, return_actions=False
                )
                # >>> n_restart x x_dim x num_categories
            else:
                X, hidden_state = maps[step], prev_hid_state

            n_fantasies = self.n_fantasy_at_design_pts[step]
            X_shape = self.n_fantasy_at_design_pts[:step][::-1] + [n_restarts, 1, x_dim, num_categories]
            X = X.reshape(*X_shape)
            # >>> num_x_{step} * x_dim * num_categories

            X4fantasize = embedder.encode(X)
            # X4fantasize = torch.nn.functional.one_hot(X.argmax(dim=-1), num_classes=num_categories).to(X.dtype)
            # X4fantasize = (X4fantasize + X - X.detach())
            # X4fantasize = embedder.encode(X4fantasize)
            # >>> num_x_{step} * x_dim
            X_returned.append(X4fantasize)
            hidden_state_returned.append(hidden_state)

            # Sample posterior
            ppd = fantasized_model.posterior(X4fantasize)
            ys = self.design_samplers[step](ppd)
            # >>> n_samples * num_x_{step} * y_dim

            X_expanded_shape = [ys.shape[0]] + [-1] * len(X_shape)
            X_expanded = X[None, ...].expand(*X_expanded_shape)
            # >>> n_samples * num_x_{step} * 1 * x_dim * num_categories
            
            # Xy = torch.cat((X_expanded, ys), dim=-1)
            # >>> n_samples * num_x_{step} * 1 * dim

            # Update previous_Xy
            # previous_Xy = Xy.reshape(-1, x_dim + y_dim)
            previous_X = X_expanded.reshape(-1, x_dim, num_categories)
            previous_y = ys.reshape(-1, y_dim)
            # >>> (n_samples * num_x_{step}) * y_dim

            # Update conditions
            fantasized_model = fantasized_model.condition_on_observations(X4fantasize, ys)

            # Update hidden state
            prev_hid_state = hidden_state[None, ...]
            prev_hid_state = prev_hid_state.expand(n_fantasies, -1, -1)
            prev_hid_state = prev_hid_state.reshape(-1, hidden_state.shape[-1])

        # Compute actions
        if use_amortized_map:
            actions, hidden_state = maps(
                x=previous_X, y=previous_y, prev_hid_state=prev_hid_state, return_actions=True
            )
        else:
            actions = maps[self.lookahead_steps]

        actions = embedder.encode(actions)
        # actions_ = torch.nn.functional.one_hot(actions.argmax(dim=-1), num_classes=num_categories).to(X.dtype)
        # actions = (actions_ + actions - actions.detach())
        # actions = embedder.encode(actions)
        
        action_shape = self.n_fantasy_at_design_pts[::-1] + [
            n_restarts,
            self.n_actions,
            x_dim,
        ]
        actions = actions.reshape(*action_shape)
        action_yis = self.action_sampler(fantasized_model.posterior(actions))
        # >> Tensor[*[n_samples]*i, n_restarts, n_actions, 1]
        
        action_yis = action_yis.squeeze(dim=-1)
        # >> Tensor[*[n_samples]*i, n_restarts, n_actions]

        # Calculate loss value
        acqf_loss = self.loss_function(actions, action_yis)

        # Calculate cost value
        single_prev_X = embedder.encode(prev_X[:, None, ...])
        # single_prev_X = torch.nn.functional.one_hot(prev_X[:, None, ...].argmax(dim=-1), num_classes=num_categories).to(X.dtype)
        # single_prev_X = (single_prev_X + prev_X[:, None, ...] - prev_X[:, None, ...].detach())
        # single_prev_X = embedder.encode(single_prev_X)
        acqf_cost = self.cost_function(
            prev_X=single_prev_X, current_X=X_returned[0]
        )
        for i in range(self.lookahead_steps - 1):
            cX = X_returned[i + 1]
            pX = X_returned[i][None, ...].expand_as(cX)
            acqf_cost = acqf_cost + self.cost_function(prev_X=pX, current_X=cX)
        for i in range(self.n_actions):
            cX = actions[..., i : i + 1, :]
            pX = X_returned[-1][None, ...].expand_as(cX)
            acqf_cost = acqf_cost + self.cost_function(prev_X=pX, current_X=cX)
        acqf_cost = acqf_cost.squeeze(dim=-1)
        
        # Reduce dimensions
        while len(acqf_loss.shape) > 1:
            acqf_loss = acqf_loss.mean(dim=0)
            acqf_cost = acqf_cost.mean(dim=0)
        # >>> batch number of x_0

        return {
            "acqf_loss": acqf_loss,
            "acqf_cost": acqf_cost,
            "X": X_returned, # [num_x_{step} * x_dim * num_categories]
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

        if n_fantasy == 1:
            sampler = PosteriorMeanSampler(
                sample_shape=n_fantasy, 
                collapse_batch_dims=True
            )
        else:
            sampler = SobolQMCNormalSampler(
                sample_shape=n_fantasy, resample=False, collapse_batch_dims=True
            )

    elif n_fantasy is not None:
        if sampler.sample_shape != torch.Size([n_fantasy]):
            raise ValueError("The sampler shape must match {n_fantasy}.")
    else:
        n_fantasy = sampler.sample_shape[0]

    return sampler, n_fantasy


class qLossFunctionTopK(nn.Module):
    """Loss function for Top-K task."""

    def __init__(self, dist_weight: float, dist_threshold: float) -> None:
        r"""Batch loss function for the task of finding top-K.

        Args:
            loss_func_hypers: hyperparameters for the
                loss function class.
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
            # >>> n_fantasy_at_design_pts x batch_size x num_actions
            # ... x num_actions

            A_distance_triu[A_distance_triu > self.dist_threshold] = self.dist_threshold
            denominator = num_actions * (num_actions - 1) / 2.0
            dist_reward = A_distance_triu.sum((-1, -2)) / denominator
            # >>> n_fantasy_at_design_pts x batch_size

        qloss = -Y - self.dist_weight * dist_reward
        # >>> n_fantasy_at_design_pts x batch_size

        return qloss


class qCostFunctionSpotlight(nn.Module):
    """Splotlight cost function."""

    def __init__(self, radius: float) -> None:
        r"""Spotlight cost function."""
        super().__init__()
        self.register_buffer("radius", torch.as_tensor(radius))

    def forward(self, prev_X: Tensor, current_X: Tensor):
        """Calculate splotlight cost.

        If distance between two points is smaller than radius,
        the cost will be zero. Otherwise, the cost will be a
        large value.

        Args:
            prev_X (Tensor): A tensor of ... x x_dim of previous X
            current_X (Tensor): A tensor of n_fantasies x ... x
                x_dim of current X

        Returns:
            Tensor: A tensor of ... x 1 cost values
        """
        diff = torch.sqrt(torch.pow(current_X - prev_X, 2).sum(-1))
        nb_idx = diff < self.radius
        diff = diff * (1 - nb_idx.float()) * 100
        return diff


class qCostFunctionL2(nn.Module):
    """L2 cost function."""

    def __init__(self) -> None:
        r"""L2 cost function."""
        super().__init__()

    def forward(self, prev_X: Tensor, current_X: Tensor):
        """Calculate L2 cost.

        Args:
            prev_X (Tensor): A tensor of ... x x_dim of previous X
            current_X (Tensor): A tensor of ... x x_dim of current X

        Returns:
            Tensor: A tensor of ... x 1 cost values
        """
        diff = torch.sqrt(torch.pow(current_X - prev_X, 2).sum(-1))
        return diff
