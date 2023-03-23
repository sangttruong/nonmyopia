from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead, warmstart_multistep
from botorch.optim import optimize_acqf
from botorch.sampling.normal import IIDNormalSampler
from botorch.acquisition.objective import LinearMCObjective, ScalarizedObjective
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from models.hentropy import qHEntropySearchTopK
from torch import Tensor
from torch.nn import Module
import random
from copy import deepcopy


class qHEntropySearchMultiStepTopK(qMultiStepLookahead):
    def __init__(
            self,
            model: Model,
            batch_size: int,
            lookahead_batch_sizes: List[int],
            num_fantasies: Optional[List[int]] = None,
            samplers: Optional[List[MCSampler]] = None,
            X_pending: Optional[Tensor] = None,
            collapse_fantasy_base_samples: bool = True
    ) -> None:
        self.batch_size = batch_size
        batch_sizes = [batch_size] + lookahead_batch_sizes

        # TODO: This objective is never really used.
        weights = torch.zeros(model.num_outputs, dtype=torch.double)
        weights[0] = 1.0

        objective = LinearMCObjective(weights=weights)
        valfunc_cls = [qHEntropySearchTopK for _ in batch_sizes]
        inner_mc_samples = [128 for bs in batch_sizes]

        valfunc_argfacs = None

        # Set samplers
        if samplers is None:
            # The batch_range is not set here and left to sampler default of (0, -2),
            # meaning that collapse_batch_dims will be applied on fantasy batch dimensions.
            # If collapse_fantasy_base_samples is False, the batch_range is updated during
            # the forward call.
            samplers: List[MCSampler] = [
                SobolQMCNormalSampler(
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
            collapse_fantasy_base_samples=collapse_fantasy_base_samples
        )
        
        
def custom_warmstart_multistep(
    acq_function: qMultiStepLookahead,
    bounds: Tensor,
    num_restarts: int,
    raw_samples: int,
    n_lookahead_steps:int,
    full_optimizer: Tensor
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
    
    input_dim = batch_initial_conditions.shape[-1]
    batch_shape, shapes, sizes = acq_function.get_split_shapes(X=batch_initial_conditions)

    # Safe copy of model
    model = deepcopy(acq_function.model)
    obj_model = model.subset_output(idcs=[0])

    #Define optimization domain
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    
    #
    aux_acq_func = PosteriorMean(model=obj_model)

    new_x, acq_value = optimize_acqf(
        acq_function=aux_acq_func,
        bounds=standard_bounds,
        q=1,
        num_restarts=5 * input_dim,
        raw_samples=100 * input_dim,
        options={},
        return_best_only=True,
    )

    i = 0
    for _ in range(sizes[0]):
        batch_initial_conditions[random_index, i, :] = new_x.clone().squeeze(0)
        i += 1

    # Fantasize objective and cost values
    sampler = IIDNormalSampler(
        num_samples=1, resample=True, collapse_batch_dims=True
    )
    posterior_new_x = model.posterior(new_x, observation_noise=True)
    fantasy_obs = sampler(posterior_new_x).squeeze(dim=0).detach()
    model = model.condition_on_observations(X=new_x, Y=fantasy_obs)

    if n_lookahead_steps > 0:
        aux_acq_func = qHEntropySearchMultiStepTopK(
            model=model,
            batch_size=1,
            lookahead_batch_sizes=[1 for _ in range(n_lookahead_steps)],
            num_fantasies=[1 for _ in range(n_lookahead_steps)],
        )
    else:
        aux_acq_func = qHEntropySearchTopK(model=model)


    new_x, acq_value = optimize_acqf(
            acq_function=aux_acq_func,
            bounds=standard_bounds,
            q=aux_acq_func.get_augmented_q_batch_size(1) if n_lookahead_steps > 0 else 1,
            num_restarts=5 * input_dim,
            raw_samples=100 * input_dim,
            options={},
            return_best_only=True,
            return_full_tree=True,
        )

    for j, size in enumerate(sizes[1:]):
        for _ in range(size):
            batch_initial_conditions[random_index, i, :] = new_x[j, :].clone()
            i += 1

    return batch_initial_conditions
    