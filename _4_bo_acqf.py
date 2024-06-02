#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List, Optional, Tuple, Type

import copy
import torch
import torch.nn as nn
from torch import Tensor
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.sampling.base import MCSampler
from botorch import settings
from botorch.models.utils.assorted import fantasize as fantasize_flag
from botorch.acquisition import (
    qExpectedImprovement,
    qKnowledgeGradient,
    qMultiStepLookahead,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
    qNegIntegratedPosteriorVariance,
)


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
        num_fantasies: Optional[List[int]] = 64,
        sampler: Optional[MCSampler] = None,
        best_f: Optional[float] = None,
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
                model=self.model, num_fantasies=num_fantasies)

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

        elif self.name == "qUCB":
            self.bo_acqf = qUpperConfidenceBound(model=self.model, sampler=sampler)

        elif self.name == "qMSL":
            self.bo_acqf = qMultiStepLookahead(
                model=self.model,
                batch_sizes=[1] * self.algo_lookahead_steps,
                num_fantasies=num_fantasies,
            )

        elif self.name == "qNIPV":
            self.bo_acqf = qNegIntegratedPosteriorVariance(
                model=self.model, mc_points=0, sampler=sampler  # TODO
            )

    def forward(
        self,
        prev_X: Tensor,
        prev_y: Tensor,
        maps: List[nn.Module],
        embedder: nn.Module = None,
        prev_cost: float = 0.0,
        **kwargs
    ) -> Dict[str, Tensor]:
        
        n_restarts = prev_X.shape[0]
        x_dim = prev_X.shape[1]
        
        actions = torch.concat(maps)
        if embedder is not None:
            actions = embedder.encode(actions)
            
        action_shape = [
            n_restarts,
            -1,
            x_dim,
        ]
        
        actions = actions.reshape(*action_shape)
        acqf_loss =  - self.bo_acqf(actions)
        # >>> batch_size

        pX = prev_X[:, None, ...]
        if self.name == "qMSL":
            acqf_cost = 0
            for cX in self.bo_acqf.get_multi_step_tree_input_representation(actions):
                acqf_cost = acqf_cost + self.cost_function(
                    prev_X=pX.expand_as(cX), current_X=cX, previous_cost=acqf_cost + prev_cost
                )
                pX = cX[None, ...]
        else:
            acqf_cost = self.cost_function(
                prev_X=pX, current_X=actions, previous_cost=prev_cost
            )
        acqf_cost = acqf_cost.squeeze(dim=-1).sum(dim=-1)
        while len(acqf_cost.shape) > 1:
            acqf_cost = acqf_cost.mean(dim=0)

        if self.name == "qMSL":
            X_returned = self.bo_acqf.get_multi_step_tree_input_representation(actions)
        else:
            X_returned = [actions]
            
        return {
            "acqf_loss": acqf_loss,
            "acqf_cost": acqf_cost,
            "X": X_returned,
            "actions": actions,
            "hidden_state": None
        }