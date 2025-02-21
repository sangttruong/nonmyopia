#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Multi-step H-Entropy Search with one-shot optimization."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from botorch import settings
from botorch.acquisition import (
    qExpectedImprovement,
    qKnowledgeGradient,
    qMultiStepLookahead,
    qNegIntegratedPosteriorVariance,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.utils.assorted import fantasize as fantasize_flag
from botorch.posteriors import Posterior
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import NormalMCSampler, SobolQMCNormalSampler
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths
from torch import Tensor


class qBOAcqf(MCAcquisitionFunction):
    """qMultiStep H-Entropy Search Class."""

    def __init__(
        self,
        name,
        model,
        loss_function_class: Type[nn.Module],
        loss_func_hypers: Dict[str, int],
        cost_function_class: Type[nn.Module],
        cost_func_hypers: Dict[str, int],
        lookahead_steps: int,
        n_actions: int,
        n_fantasy_at_design_pts: Optional[List[int]] = [64],
        sampler: Optional[MCSampler] = None,
        best_f: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(model=model)
        self.name = name
        self.model = model
        self.algo_lookahead_steps = lookahead_steps
        self.n_actions = n_actions
        self.cost_function = cost_function_class(**cost_func_hypers)
        self.loss_function = loss_function_class(**loss_func_hypers)

        if self.name == "qKG":
            self.bo_acqf = qKnowledgeGradient(
                model=self.model,
                num_fantasies=n_fantasy_at_design_pts[0],
                sampler=sampler,
            )

        elif self.name == "qEI":
            self.bo_acqf = qExpectedImprovement(
                model=self.model,
                best_f=best_f,
                sampler=sampler,
            )

        elif self.name == "qPI":
            self.bo_acqf = qProbabilityOfImprovement(
                model=self.model,
                best_f=best_f,
                sampler=sampler,
            )

        elif self.name == "qSR":
            self.bo_acqf = qSimpleRegret(model=self.model, sampler=sampler)

        elif self.name.startswith("qUCB"):
            if self.name == "qUCB":
                beta = 0.1
            else:
                beta = float(self.name.split("-")[-1])
            self.bo_acqf = qUpperConfidenceBound(
                model=self.model, beta=beta, sampler=sampler
            )

        elif self.name == "qMSL":
            self.bo_acqf = qMultiStepLookahead(
                model=self.model,
                batch_sizes=[1] * self.algo_lookahead_steps,
                num_fantasies=n_fantasy_at_design_pts,
            )

        elif self.name == "qNIPV":
            self.bo_acqf = qNegIntegratedPosteriorVariance(
                model=self.model, mc_points=0, sampler=sampler
            )

        else:
            raise NotImplementedError

    def forward(
        self,
        prev_X: Tensor,
        prev_y: Tensor,
        maps: List[nn.Module],
        embedder: nn.Module = None,
        prev_cost: float = 0.0,
        **kwargs,
    ) -> Dict[str, Tensor]:

        n_restarts = prev_X.shape[0]
        x_dim = prev_X.shape[1]

        actions = torch.concat(maps)
        pX = prev_X[:, None, ...]
        if embedder is not None:
            pX = embedder.encode(pX)
            actions = embedder.encode(actions)

        action_shape = [
            n_restarts,
            -1,
            x_dim,
        ]
        actions = actions.reshape(*action_shape)
        acqf_loss = -self.bo_acqf(actions)
        # >>> batch_size

        if self.name == "qMSL":
            acqf_cost = 0
            for cX in self.bo_acqf.get_multi_step_tree_input_representation(actions):
                acqf_cost = acqf_cost + self.cost_function(
                    prev_X=pX.expand_as(cX),
                    current_X=cX,
                    previous_cost=acqf_cost + prev_cost,
                )
                pX = cX[None, ...]
        elif self.name == "qKG":
            cX = actions[..., : -self.bo_acqf.num_fantasies, :]
            acqf_cost = self.cost_function(
                prev_X=pX.expand_as(cX), current_X=cX, previous_cost=prev_cost
            )
            pX = cX
            cX = actions[..., -self.bo_acqf.num_fantasies :, :]
            acqf_cost = acqf_cost + self.cost_function(
                prev_X=pX.expand_as(cX),
                current_X=cX,
                previous_cost=acqf_cost + prev_cost,
            )
        else:
            acqf_cost = self.cost_function(
                prev_X=pX.expand_as(actions), current_X=actions, previous_cost=prev_cost
            )
        acqf_cost = acqf_cost.squeeze(dim=-1).sum(dim=-1)
        while len(acqf_cost.shape) > 1:
            acqf_cost = acqf_cost.mean(dim=0)

        if self.name == "qMSL":
            X_returned = self.bo_acqf.get_multi_step_tree_input_representation(actions)
        elif self.name == "qKG":
            X_returned = [self.bo_acqf.extract_candidates(actions)]
        else:
            X_returned = [actions]

        return {
            "acqf_loss": acqf_loss,
            "acqf_cost": acqf_cost,
            "X": X_returned,
            "actions": actions,
            "hidden_state": None,
        }


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
        enable_ts: Optional[bool] = False,
        **kwargs,
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
        self.enable_ts = enable_ts
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

    def dump_model(self, f: Optional[nn.Module] = None):
        """Dump model."""
        self._model = copy.deepcopy(self.model)
        if self.enable_ts:
            if f is None:
                # Draw new f ~ p(f|D)
                self.f = draw_matheron_paths(self.model, torch.Size([1]))
            else:
                self.f = f

    def clean_dump_model(self):
        """Clean dump model."""
        del self._model
        torch.cuda.empty_cache()
        self._model = None
        if self.enable_ts:
            del self.f
            self.f = None

    def forward(
        self,
        prev_X: Tensor,
        prev_y: Tensor,
        prev_hid_state: Tensor,
        maps: List[nn.Module],
        embedder: nn.Module = None,
        prev_cost: float = 0.0,
        enable_noise: bool = True,
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

        if not self.enable_ts:
            if self._model is None:
                fantasized_model = self.model
            else:
                fantasized_model = self._model

        X_returned = []
        hidden_state_returned = []
        for step in range(self.algo_lookahead_steps):
            # condition on X[step], then sample, then condition on (x,prev_X y)
            if use_amortized_map:
                X, hidden_state = maps(
                    x=previous_X,
                    y=previous_y,
                    prev_hid_state=prev_hid_state,
                    return_actions=False,
                    enable_noise=enable_noise,
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
                # Cat ==> Con
                X = embedder.encode(X)
                # >>> num_x_{step} * x_dim

            X_returned.append(X)
            hidden_state_returned.append(hidden_state)

            # Sample posterior
            if self.enable_ts:
                ys = self.f(X.squeeze(dim=list(range(X.dim() - 3)))).unsqueeze(0)
            else:
                with fantasize_flag():
                    with settings.propagate_grads(False):
                        ppd = fantasized_model.posterior(X)
                    ys = self.design_samplers[step](ppd).to(X)
                    # >>> n_samples * num_x_{step} * y_dim

                    # Update conditions
                    fantasized_model = fantasized_model.condition_on_observations(
                        X=fantasized_model.transform_inputs(X), Y=ys
                    )

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
            # Cat ==> Con
            actions = embedder.encode(actions)
        action_shape = self.n_fantasy_at_design_pts[::-1] + [
            n_restarts,
            self.n_actions,
            x_dim,
        ]
        actions = actions.reshape(*action_shape)
        if self.enable_ts:
            action_yis = self.f(
                actions.squeeze(dim=list(range(actions.dim() - 3)))
            ).unsqueeze(0)
        else:
            action_yis = self.action_sampler(
                fantasized_model.posterior(actions)
            ).squeeze(dim=-1)
        # >> Tensor[*[n_samples]*i, n_restarts, n_actions]

        # Calculate loss value
        acqf_loss = self.loss_function(actions, action_yis)

        # Calculate cost value
        first_prev_X = prev_X[:, None, ...]
        if embedder is not None:
            # Cat ==> Con
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
            cX = actions[..., i : i + 1, :]
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


class PosteriorMeanSampler(NormalMCSampler):
    r"""Sampler for MC base samples using iid N(0,1) samples.

    Example:
        >>> sampler = IIDNormalSampler(1000, seed=1234)
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def _construct_base_samples(self, posterior: Posterior) -> None:
        r"""Generate iid `N(0,1)` base samples (if necessary).

        This function will generate a new set of base samples and set the
        `base_samples` buffer if one of the following is true:

        - the MCSampler has no `base_samples` attribute.
        - the output of `_get_collapsed_shape` does not agree with the shape of
            `self.base_samples`.

        Args:
            posterior: The Posterior for which to generate base samples.
        """
        target_shape = self._get_collapsed_shape(posterior=posterior)
        if self.base_samples is None or self.base_samples.shape != target_shape:
            base_samples = torch.zeros(
                target_shape, device=posterior.device, dtype=posterior.dtype
            )
            self.register_buffer("base_samples", base_samples)
        if self.base_samples.device != posterior.device:
            self.to(device=posterior.device)  # pragma: nocover
        if self.base_samples.dtype != posterior.dtype:
            self.to(dtype=posterior.dtype)


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
                sample_shape=n_fantasy, collapse_batch_dims=True
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

    def __init__(
        self,
        dist_weight: float,
        dist_threshold: float,
    ) -> None:
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


class qCostFunction(nn.Module):
    """Splotlight cost function."""

    def __init__(
        self,
        radius: float,
        k: float = 1,
        max_noise: float = 1e-5,
        p_norm: float = 2.0,
        discount: float = 0.0,
        discount_threshold: float = -1.0,
    ) -> None:
        r"""Spotlight cost function."""
        super().__init__()
        self.register_buffer("radius", torch.as_tensor(radius))
        self.k = k
        self.max_noise = max_noise
        self.p_norm = p_norm
        self.discount = discount
        self.discount_threshold = discount_threshold

    def forward(
        self, prev_X: Tensor, current_X: Tensor, previous_cost: Tensor = None
    ) -> Tensor:
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
        diff = torch.cdist(current_X, prev_X, p=self.p_norm)
        diff = (
            torch.max(self.k * (diff - self.radius), torch.zeros_like(diff))
            + torch.randn_like(diff) * self.max_noise
        )
        if self.discount > 0.0:
            diff = diff * (
                1
                - self.discount
                * (previous_cost + diff > self.discount_threshold).float()
            )

        return diff


class qCostFunctionEditDistance(nn.Module):
    """Edit Distance cost function."""

    def __init__(
        self,
        radius: float,
        discount: float = 0.0,
        discount_threshold: float = -1.0,
        previous_loss: Optional[nn.Module] = None,
    ) -> None:
        r"""Edit Distance cost function."""
        super().__init__()
        self.register_buffer("radius", torch.as_tensor(radius))

    def forward(
        self, prev_X: Tensor, current_X: Tensor, previous_cost: Tensor = None
    ) -> Tensor:
        """Calculate EditDistance cost.

        If number of edit points is radius,
        the cost will be zero. Otherwise, the cost will be a
        number of edit points.

        Args:
            prev_X (Tensor): A tensor of ... x x_dim of previous X
            current_X (Tensor): A tensor of n_fantasies x ... x
                x_dim of current X

        Returns:
            Tensor: A tensor of ... x 1 cost values
        """
        diff = self.editdistance(prev_X, current_X)
        nb_idx = diff <= self.radius
        diff = diff * (1 - nb_idx.float()) * 100
        return diff

    def editdistance(self, prev_X: Tensor, current_X: Tensor):
        diff = prev_X[..., None, :] - current_X[..., None, :, :]
        nb_idx = torch.abs(diff) >= 1e-5
        diff = torch.abs(diff) * nb_idx.float()
        nb_idx2 = diff > 0
        diff = 1 - (1 - diff) * (1 - nb_idx2.float())
        diff1 = torch.squeeze(torch.sum(diff, dim=-1).to(dtype=torch.int64), dim=-1)

        return diff1
