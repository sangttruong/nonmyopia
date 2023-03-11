#!/usr/bin/env python
""""""

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Type, Union

import torch
from torch import Tensor

from botorch import settings
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.monte_carlo import MCAcquisitionFunction, qSimpleRegret
from botorch.acquisition.objective import (
    AcquisitionObjective,
    MCAcquisitionObjective,
    ScalarizedObjective,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)

__author__ = ""
__copyright__ = "Copyright 2022, Stanford University"


def _get_value_function(
    model: Model,
    objective: Optional[Union[MCAcquisitionObjective,
                              ScalarizedObjective]] = None,
    sampler: Optional[MCSampler] = None,
    project: Optional[Callable[[Tensor], Tensor]] = None,
    valfunc_cls: Optional[Type[AcquisitionFunction]] = None,
    valfunc_argfac: Optional[Callable[[Model, Dict[str, Any]], Any]] = None,
) -> AcquisitionFunction:
    r"""Construct value function (i.e. inner acquisition function)."""
    if valfunc_cls is not None:
        common_kwargs: Dict[str, Any] = {
            "model": model, "objective": objective}
        if issubclass(valfunc_cls, MCAcquisitionFunction):
            common_kwargs["sampler"] = sampler
        kwargs = valfunc_argfac(
            model=model) if valfunc_argfac is not None else {}
        base_value_function = valfunc_cls(**common_kwargs, **kwargs)
    else:
        if isinstance(objective, MCAcquisitionObjective):
            base_value_function = qSimpleRegret(
                model=model, sampler=sampler, objective=objective
            )
        else:
            base_value_function = PosteriorMean(
                model=model, objective=objective)

    if project is None:
        return base_value_function


class MyqKnowledgeGradient(MCAcquisitionFunction, OneShotAcquisitionFunction):
    r"""Batch Knowledge Gradient using one-shot optimization.

    This computes the batch Knowledge Gradient using fantasies for the outer
    expectation and either the model posterior mean or MC-sampling for the inner
    expectation.

    In addition to the design variables, the input `X` also includes variables
    for the optimal designs for each of the fantasy models. For a fixed number
    of fantasies, all parts of `X` can be optimized in a "one-shot" fashion.
    """

    def __init__(
        self,
        model: Model,
        config,
        sampler: Optional[MCSampler] = None,
        objective: Optional[AcquisitionObjective] = None,
        inner_sampler: Optional[MCSampler] = None,
        X_pending: Optional[Tensor] = None,
        current_value: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        r"""q-Knowledge Gradient (one-shot optimization).

        Args:
            model: A fitted model. Must support fantasizing.
            num_fantasies: The number of fantasy points to use. More fantasy
                points result in a better approximation, at the expense of
                memory and wall time. Unused if `sampler` is specified.
            sampler: The sampler used to sample fantasy observations. Optional
                if `num_fantasies` is specified.
            objective: The objective under which the samples are evaluated. If
                `None` or a ScalarizedObjective, then the analytic posterior mean
                is used, otherwise the objective is MC-evaluated (using
                inner_sampler).
            inner_sampler: The sampler used for inner sampling. Ignored if the
                objective is `None` or a ScalarizedObjective.
            X_pending: A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.
            current_value: The current value, i.e. the expected best objective
                given the observed points `D`. If omitted, forward will not
                return the actual KG value, but the expected best objective
                given the data set `D u X`.
        """
        num_fantasies = config.num_outer_mc

        if sampler is None:
            if num_fantasies is None:
                raise ValueError(
                    "Must specify `num_fantasies` if no `sampler` is provided."
                )
            # base samples should be fixed for joint optimization over X, X_fantasies
            sampler = SobolQMCNormalSampler(
                num_samples=num_fantasies, resample=False, collapse_batch_dims=True
            )
        elif num_fantasies is not None:
            if sampler.sample_shape != torch.Size([num_fantasies]):
                raise ValueError(
                    f"The sampler shape must match num_fantasies={num_fantasies}."
                )
        else:
            num_fantasies = sampler.sample_shape[0]
        super(MCAcquisitionFunction, self).__init__(model=model)
        # if not explicitly specified, we use the posterior mean for linear objs
        if isinstance(objective, MCAcquisitionObjective) and inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(
                num_samples=128, resample=False, collapse_batch_dims=True
            )
        if objective is None and model.num_outputs != 1:
            raise UnsupportedError(
                "Must specify an objective when using a multi-output model."
            )
        self.sampler = sampler
        self.objective = objective
        self.set_X_pending(X_pending)
        self.inner_sampler = inner_sampler
        self.num_fantasies = num_fantasies
        self.current_value = current_value

    # @t_batch_mode_transform()
    def forward(self, X_actual: Tensor, X_fantasies: Tensor) -> Tensor:
        r"""Evaluate qKnowledgeGradient on the candidate set `X`.

        Args:
            X: A `b x (q + num_fantasies) x d` Tensor with `b` t-batches of
                `q + num_fantasies` design points each. We split this X tensor
                into two parts in the `q` dimension (`dim=-2`). The first `q`
                are the q-batch of design points and the last num_fantasies are
                the current solutions of the inner optimization problem.

                `X_fantasies = X[..., -num_fantasies:, :]`
                `X_fantasies.shape = b x num_fantasies x d`

                `X_actual = X[..., :-num_fantasies, :]`
                `X_actual.shape = b x q x d`

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KG value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X_actual[b]`, and `X_fantasies[b, : ]` must be
                maximized at fixed `X_actual[b]`.
        """
        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X_actual, sampler=self.sampler, observation_noise=True
        )

        # get the value function
        value_function = _get_value_function(
            model=fantasy_model, objective=self.objective, sampler=self.inner_sampler
        )

        # make sure to propagate gradients to the fantasy model train inputs
        with settings.propagate_grads(True):
            X = torch.permute(X_fantasies, [1, 0, 2, 3])  # NOTE: permute?
            X = X[:, :, 0:1, :]  # Only optimize first action variable
            values = value_function(X=X)  # num_fantasies x b

        if self.current_value is not None:
            values = values - self.current_value

        # return average over the fantasy samples
        return values.mean(dim=0)

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimization.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimization (including variables
            parameterizing the fantasy solutions).
        """
        return q + self.num_fantasies

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""We only return X as the set of candidates post-optimization.

        Args:
            X_full: A `b x (q + num_fantasies) x d`-dim Tensor with `b`
                t-batches of `q + num_fantasies` design points each.

        Returns:
            A `b x q x d`-dim Tensor with `b` t-batches of `q` design points each.
        """
        return X_full[..., : -self.num_fantasies, :]


def initialize_action_tensor_kg(config):
    """Initialize and return X_action tensor."""
    bounds_diff_design = config.bounds_design[1] - config.bounds_design[0]

    X_action = config.bounds_design[0] + bounds_diff_design * torch.rand(
        config.num_restarts,
        config.num_outer_mc,
        2,
        config.num_dim_design,
    )

    # Initialize actions consistently across fantasies
    X_action[:, 1:, :, :] = X_action[:, 0:1, :, :]

    X_action.requires_grad_(True)

    return X_action
