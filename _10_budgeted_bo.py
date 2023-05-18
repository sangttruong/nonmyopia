#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Callable

import torch
import random

from copy import copy, deepcopy
from botorch.acquisition.multi_step_lookahead import (
    qMultiStepLookahead,
    warmstart_multistep,
)
from botorch.acquisition.objective import (
    LinearMCObjective,
    MCAcquisitionObjective,
    ScalarizedObjective,
)
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler, IIDNormalSampler
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction, PosteriorMean
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.utils.objective import soft_eval_constraint
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from botorch.generation.gen import get_best_candidates

from torch import Tensor
from torch.nn import Module
from torch.distributions import Normal
from botorch.optim.optimize import optimize_acqf

from _6_samplers import PosteriorMeanSampler


class BudgetedExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Analytic Budgeted Expected Improvement.

    Computes the analytic expected improvement weighted by a probability of
    satisfying a budget constraint. The objective and (log-) cost are assumed
    to be independent and have Gaussian posterior distributions. Only supports
    the case `q=1`. The model should be two-outcome, with the first output
    corresponding to the objective and the second output corresponding to the
    log-cost.
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        budget: Union[float, Tensor],
        maximize: bool = True,
        objective: Optional[ScalarizedObjective] = None,
    ) -> None:
        r"""Analytic Budgeted Expected Improvement.
        Args:
            model: A fitted two-outcome model, where the first output corresponds
                to the objective and the second one to the log-cost.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            budget: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the budget constraint.
            maximize: If True, consider the problem a maximization problem.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.objective = None
        self.maximize = maximize
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("budget", torch.as_tensor(budget))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.
        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
            design points `X`.
        """
        posterior = self._get_posterior(X=X)
        means = posterior.mean  # (b) x 2
        sigmas = posterior.variance.sqrt().clamp_min(1e-6)  # (b) x 2

        # (b) x 1
        mean_obj = means[..., 0]
        sigma_obj = sigmas[..., 0]
        u = (mean_obj - self.best_f) / sigma_obj

        if not self.maximize:
            u = -u
        standard_normal = Normal(
            torch.zeros(1, device=u.device, dtype=u.dtype),
            torch.ones(1, device=u.device, dtype=u.dtype),
        )
        pdf_u = torch.exp(standard_normal.log_prob(u))
        cdf_u = standard_normal.cdf(u)
        ei = sigma_obj * (pdf_u + u * cdf_u)  # (b) x 1
        # (b) x 1
        prob_feas = self._compute_prob_feas(means=means[..., 1], sigmas=sigmas[..., 1])
        bc_ei = ei.mul(prob_feas)  # (b) x 1
        return bc_ei.squeeze(dim=-1)

    def _compute_prob_feas(self, means: Tensor, sigmas: Tensor) -> Tensor:
        r"""Compute feasibility probability for each batch of X.
        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
            means: A `(b) x 1`-dim Tensor of means.
            sigmas: A `(b) x 1`-dim Tensor of standard deviations.
        Returns:
            A `(b) x 1`-dim tensor of feasibility probabilities.
        """
        standard_normal = Normal(
            torch.zeros(1, device=means.device, dtype=means.dtype),
            torch.ones(1, device=means.device, dtype=means.dtype),
            validate_args=True,
        )
        prob_feas = standard_normal.cdf(
            (torch.log(self.budget.clamp_min(1e-6)) - means) / sigmas
        )
        prob_feas = torch.where(
            self.budget > 1e-6,
            prob_feas,
            torch.zeros(1, device=means.device, dtype=means.dtype),
        )
        return prob_feas


class qBudgetedExpectedImprovement(MCAcquisitionFunction):
    r"""Batch Budget-Constrained Expected Improvement.

    Computes the expected improvement weighted by a probability of satisfying
    a budget constraint. The model should be two-outcome, with the first output
    corresponding to the objective and the second output corresponding to the
    log-cost.
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        budget: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""Batch Budgeted Expected Improvement.
        Args:
            model: A fitted two-outcome model, where the first output corresponds
                to the objective and the second one to the log-cost.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            log_budget: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the budget constraint.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)`.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(MCAcquisitionFunction, self).__init__(model=model)
        self.sampler = sampler
        self.objective = None
        self.X_pending = X_pending
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("budget", torch.as_tensor(budget))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
            design points `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        improvements = (samples[..., 0] - self.best_f).clamp_min(0)
        max_improvement = improvements.max(dim=-1, keepdim=True)[0]
        sum_costs = torch.exp(samples[..., 1]).sum(dim=-1, keepdim=True)
        smooth_feas_ind = soft_eval_constraint(lhs=sum_costs - self.budget)
        bc_ei = torch.mul(max_improvement, smooth_feas_ind).mean(dim=0)
        return bc_ei.squeeze(dim=-1)


class BudgetedMultiStepExpectedImprovement(qMultiStepLookahead):
    r"""Budget-Constrained Multi-Step Look-Ahead Expected Improvement (one-shot optimization)."""

    def __init__(
        self,
        model: Model,
        budget_plus_cumulative_cost: Union[float, Tensor],
        batch_size: int,
        lookahead_batch_sizes: List[int],
        num_fantasies: Optional[List[int]] = None,
        samplers: Optional[List[MCSampler]] = None,
        X_pending: Optional[Tensor] = None,
        collapse_fantasy_base_samples: bool = True,
    ) -> None:
        r"""Budgeted Multi-Step Expected Improvement.

        Args:
            model: A fitted two-output model, where the first output corresponds to the
                objective, and the second one to the log-cost.
            budget: A value determining the budget constraint.
            batch_size: Batch size of the current step.
            lookahead_batch_sizes: A list `[q_1, ..., q_k]` containing the batch sizes for the
            `k` look-ahead steps.
            num_fantasies: A list `[f_1, ..., f_k]` containing the number of fantasy
                for the `k` look-ahead steps.
            samplers: A list of MCSampler objects to be used for sampling fantasies in
                each stage.
            X_pending: A `m x d`-dim Tensor of `m` design points that have points that
                have been submitted for function evaluation but have not yet been
                evaluated. Concatenated into `X` upon forward call. Copied and set to
                have no gradient.
            collapse_fantasy_base_samples: If True, collapse_batch_dims of the Samplers
                will be applied on fantasy batch dimensions as well, meaning that base
                samples are the same in all subtrees starting from the same level.
        """
        self.budget_plus_cumulative_cost = budget_plus_cumulative_cost
        self.batch_size = batch_size
        batch_sizes = [batch_size] + lookahead_batch_sizes

        # TODO: This objective is never really used.
        weights = torch.zeros(model.num_outputs, dtype=torch.double)
        weights[0] = 1.0

        use_mc_val_funcs = any(bs != 1 for bs in batch_sizes)

        if use_mc_val_funcs:
            objective = LinearMCObjective(weights=weights)

            valfunc_cls = [qBudgetedExpectedImprovement for _ in batch_sizes]

            inner_mc_samples = [128 for bs in batch_sizes]
        else:
            objective = ScalarizedObjective(weights=weights)

            valfunc_cls = [BudgetedExpectedImprovement for _ in batch_sizes]

            inner_mc_samples = None

        valfunc_argfacs = [
            budgeted_ei_argfac(
                budget_plus_cumulative_cost=self.budget_plus_cumulative_cost
            )
            for _ in batch_sizes
        ]

        # Set samplers
        if samplers is None:
            # The batch_range is not set here and left to sampler default of (0, -2),
            # meaning that collapse_batch_dims will be applied on fantasy batch dimensions.
            # If collapse_fantasy_base_samples is False, the batch_range is updated during
            # the forward call.
            samplers: List[MCSampler] = [
                PosteriorMeanSampler(collapse_batch_dims=True)
                if nf == 1
                else SobolQMCNormalSampler(
                    sample_shape=nf, resample=False, collapse_batch_dims=True
                )
                for nf in num_fantasies
            ]

        super().__init__(
            model=model,
            batch_sizes=lookahead_batch_sizes,
            samplers=samplers,
            valfunc_cls=valfunc_cls,
            valfunc_argfacs=valfunc_argfacs,
            objective=objective,
            inner_mc_samples=inner_mc_samples,
            X_pending=X_pending,
            collapse_fantasy_base_samples=collapse_fantasy_base_samples,
        )


class ExpectedImprovementPerUnitOfCost(AnalyticAcquisitionFunction):
    r"""Expected Improvement Per Unit of Cost (analytic)."""

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        cost_exponent: Union[float, Tensor] = 1.0,
        maximize: bool = True,
        objective: Optional[ScalarizedObjective] = None,
    ) -> None:
        r"""Single-outcome Expected Improvement (analytic).
        Args:
            model: A fitted two-outcome model, where the first output corresponds
                to the objective and the second one to the log-cost.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            maximize: If True, consider the problem a maximization problem.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.objective = None
        self.maximize = maximize
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("cost_exponent", torch.as_tensor(cost_exponent))

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement Per Unit of Cost on the candidate set X.
        Args:
            X: A `b1 x ... bk x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.
        Returns:
            A `b1 x ... bk`-dim tensor of Expected Improvement Per Unit of Cost values
            at the given design points `X`.
        """
        posterior = self._get_posterior(X=X)
        means = posterior.mean  # (b) x 2
        vars = posterior.variance.clamp_min(1e-6)  # (b) x 2
        stds = vars.sqrt()

        # (b) x 1
        mean_obj = means[..., 0]
        std_obj = stds[..., 0]
        u = (mean_obj - self.best_f) / std_obj

        if not self.maximize:
            u = -u
        standard_normal = Normal(
            torch.zeros(1, device=u.device, dtype=u.dtype),
            torch.ones(1, device=u.device, dtype=u.dtype),
        )
        pdf_u = torch.exp(standard_normal.log_prob(u))
        cdf_u = standard_normal.cdf(u)
        ei = std_obj * (pdf_u + u * cdf_u)  # (b) x 1
        # (b) x 1
        eic = torch.exp(
            -(self.cost_exponent * means[..., 1])
            + 0.5 * (torch.square(self.cost_exponent) * vars[..., 1])
        )
        ei_puc = ei.mul(eic)  # (b) x 1
        return ei_puc.squeeze(dim=-1)


class budgeted_ei_argfac(Module):
    r"""Extract the best observed value and reamaining budget from the model."""

    def __init__(self, budget_plus_cumulative_cost: Union[float, Tensor]) -> None:
        super().__init__()
        self.budget_plus_cumulative_cost = budget_plus_cumulative_cost

    def forward(self, model: Model, X: Tensor) -> Dict[str, Any]:
        y = torch.transpose(model.train_targets, -2, -1)
        y_original_scale = model.outcome_transform.untransform(y)[0]
        obj_vals = y_original_scale[..., 0]
        log_costs = y_original_scale[..., 1]
        costs = torch.exp(log_costs)
        current_budget = self.budget_plus_cumulative_cost - costs.sum(
            dim=-1, keepdim=True
        )

        params = {
            "best_f": obj_vals.max(dim=-1, keepdim=True).values,
            "budget": current_budget,
        }
        return params


def custom_warmstart_multistep(
    acq_function: qMultiStepLookahead,
    bounds: Tensor,
    num_restarts: int,
    raw_samples: int,
    full_optimizer: Tensor,
    algo_params: Dict,
) -> Tensor:
    batch_initial_conditions = warmstart_multistep(
        acq_function=acq_function,
        bounds=bounds,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        full_optimizer=full_optimizer,
    )

    n_initial_points = batch_initial_conditions.shape[0]
    random_index = random.randrange(n_initial_points)
    print(random_index)
    input_dim = batch_initial_conditions.shape[-1]
    batch_shape, shapes, sizes = acq_function.get_split_shapes(
        X=batch_initial_conditions
    )

    # Safe copy of model
    model = deepcopy(acq_function.model)
    obj_model = model.subset_output(idcs=[0])

    # Define optimization domain
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])

    #
    aux_acq_func = PosteriorMean(model=obj_model)

    new_x, acq_value = optimize_acqf(
        acq_function=aux_acq_func,
        bounds=standard_bounds,
        q=1,
        num_restarts=5 * input_dim,
        raw_samples=100 * input_dim,
        options={
            "batch_limit": 5,
            "maxiter": 100,
            "method": "L-BFGS-B",
        },
        return_best_only=True,
    )

    i = 0
    for _ in range(sizes[0]):
        batch_initial_conditions[random_index, i, :] = new_x.clone().squeeze(0)
        i += 1

    # Fantasize objective and cost values
    sampler = IIDNormalSampler(num_samples=1, resample=True, collapse_batch_dims=True)
    posterior_new_x = model.posterior(new_x, observation_noise=True)
    fantasy_obs = sampler(posterior_new_x).squeeze(dim=0).detach()
    fantasy_cost = torch.exp(fantasy_obs[0, 1]).item()
    model = model.condition_on_observations(X=new_x, Y=fantasy_obs)

    n_lookahead_steps = len(algo_params.get("lookahead_n_fantasies")) - 1

    if n_lookahead_steps > 0:
        aux_acq_func = BudgetedMultiStepExpectedImprovement(
            model=model,
            budget_plus_cumulative_cost=algo_params.get(
                "current_budget_plus_cumulative_cost"
            )
            - fantasy_cost,
            batch_size=1,
            lookahead_batch_sizes=[1 for _ in range(n_lookahead_steps)],
            num_fantasies=[1 for _ in range(n_lookahead_steps)],
        )
    else:
        y = torch.transpose(model.train_targets, -2, -1)
        y_original_scale = model.outcome_transform.untransform(y)[0]
        obj_vals = y_original_scale[..., 0]
        best_f = torch.max(obj_vals).item()

        aux_acq_func = BudgetedExpectedImprovement(
            model=model,
            best_f=best_f,
            budget=algo_params.get("current_budget") - fantasy_cost,
        )

    new_x, acq_value = optimize_acqf(
        acq_function=aux_acq_func,
        bounds=standard_bounds,
        q=aux_acq_func.get_augmented_q_batch_size(1) if n_lookahead_steps > 0 else 1,
        num_restarts=5 * input_dim,
        raw_samples=100 * input_dim,
        options={
            "batch_limit": 1,
            "maxiter": 100,
            "method": "L-BFGS-B",
        },
        return_best_only=True,
        return_full_tree=True,
    )

    for j, size in enumerate(sizes[1:]):
        for _ in range(size):
            batch_initial_conditions[random_index, i, :] = new_x[j, :].clone()
            i += 1

    return batch_initial_conditions


def fantasize_costs(
    algo: str,
    model: Model,
    n_steps: int,
    budget_left: float,
    init_budget: float,
    input_dim: int,
):
    """
    Fantasizes the observed costs when following a specified sampling
    policy for a given number of steps.
    """
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])

    fantasy_costs = []
    fantasy_optimizers = []

    if algo == "EI-PUC_CC":
        for _ in range(n_steps):
            # Acquisition function
            y = torch.transpose(model.train_targets, -2, -1)
            y_original_scale = model.outcome_transform.untransform(y)[0]
            obj_vals = y_original_scale[..., 0]
            best_f = torch.max(obj_vals).item()
            cost_exponent = budget_left / init_budget

            aux_acq_func = ExpectedImprovementPerUnitOfCost(
                model=model,
                best_f=best_f,
                cost_exponent=cost_exponent,
            )

            # Get new point
            new_x, acq_value = optimize_acqf(
                acq_function=aux_acq_func,
                bounds=standard_bounds,
                q=1,
                num_restarts=5 * input_dim,
                raw_samples=50 * input_dim,
                options={
                    "batch_limit": 5,
                    "maxiter": 100,
                    "nonnegative": True,
                    "method": "L-BFGS-B",
                },
                return_best_only=True,
            )

            fantasy_optimizers.append(new_x.clone())

            # Fantasize objective and cost values
            sampler = IIDNormalSampler(
                num_samples=1, resample=True, collapse_batch_dims=True
            )
            posterior_new_x = model.posterior(new_x, observation_noise=True)
            fantasy_obs = sampler(posterior_new_x).squeeze(dim=0).detach()
            fantasy_costs.append(torch.exp(fantasy_obs[0, 1]).item())
            model = model.condition_on_observations(X=new_x, Y=fantasy_obs)

            # Update remaining budget
            budget_left -= fantasy_costs[-1]

    print("Fantasy costs:")
    fantasy_costs = torch.tensor(fantasy_costs)
    print(fantasy_costs)
    return fantasy_costs, fantasy_optimizers


def get_suggested_budget(
    strategy: str,
    refill_until_lower_bound_is_reached: bool,
    budget_left: float,
    model: Model,
    n_lookahead_steps: int,
    X: Tensor,
    objective_X: Tensor,
    cost_X: Tensor,
    init_budget: float,
    previous_budget: Optional[float] = None,
    lower_bound: Optional[float] = None,
):
    """
    Computes the suggested budget to be used by the budgeted multi-step
    expected improvement acquisition function.
    """
    if (
        refill_until_lower_bound_is_reached
        and (lower_bound is not None)
        and (previous_budget - cost_X[-1] > lower_bound)
    ):
        suggested_budget = previous_budget - cost_X[-1].item()
        return suggested_budget, lower_bound

    if strategy == "fantasy_costs_from_aux_policy":
        # Fantasize the observed costs following the auxiliary acquisition function
        fantasy_costs, fantasy_optimizers = fantasize_costs(
            algo="EI-PUC_CC",
            model=deepcopy(model),
            n_steps=n_lookahead_steps,
            budget_left=copy(budget_left),
            init_budget=init_budget,
            input_dim=X.shape[-1],
        )

        # Suggested budget is the minimum between the sum of the fantasy costs
        # and the true remaining budget.
        suggested_budget = fantasy_costs.sum().item()
        lower_bound = fantasy_costs.min().item()
    suggested_budget = min(suggested_budget, budget_left)
    return suggested_budget, lower_bound


def evaluate_obj_and_cost_at_X(
    X: Tensor,
    objective_function: Optional[Callable],
    cost_function: Optional[Callable],
    objective_cost_function: Optional[Callable],
) -> Tensor:
    if (objective_cost_function is None) and (
        objective_function is None or cost_function is None
    ):
        raise RuntimeError(
            "Both the objective and cost functions must be passed as inputs."
        )

    if objective_cost_function is not None:
        objective_X, cost_X = objective_cost_function(X)
    else:
        objective_X = objective_function(X)
        cost_X = cost_function(X)

    return objective_X, cost_X


def optimize_acqf_and_get_suggested_point(
    acq_func: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
    algo_params: Dict,
) -> Tensor:
    """Optimizes the acquisition function, and returns the candidate solution."""
    is_ms = isinstance(acq_func, qMultiStepLookahead)
    input_dim = bounds.shape[1]
    q = acq_func.get_augmented_q_batch_size(batch_size) if is_ms else batch_size
    raw_samples = 200 * input_dim * batch_size
    num_restarts = 10 * input_dim * batch_size

    # if is_ms:
    # raw_samples *= (len(algo_params.get("lookahead_n_fantasies")) + 1)
    # num_restarts *=  (len(algo_params.get("lookahead_n_fantasies")) + 1)

    if algo_params.get("suggested_x_full_tree") is not None:
        batch_initial_conditions = custom_warmstart_multistep(
            acq_function=acq_func,
            bounds=bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            full_optimizer=algo_params.get("suggested_x_full_tree"),
            algo_params=algo_params,
        )
    else:
        batch_initial_conditions = None

    candidates, acq_values = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={
            "batch_limit": 2,
            "maxiter": 200,
            "nonnegative": True,
            "method": "L-BFGS-B",
        },
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=False,
        return_full_tree=is_ms,
    )

    candidates = candidates.detach()
    if is_ms:
        # save all tree variables for multi-step initialization
        algo_params["suggested_x_full_tree"] = candidates.clone()
        candidates = acq_func.extract_candidates(candidates)

    acq_values_sorted, indices = torch.sort(acq_values.squeeze(), descending=True)
    print("Acquisition values:")
    print(acq_values_sorted)
    print("Candidates:")
    print(candidates[indices].squeeze())
    print(candidates.squeeze())

    new_x = get_best_candidates(batch_candidates=candidates, batch_values=acq_values)
    return new_x
