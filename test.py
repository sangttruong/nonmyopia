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
from botorch.models import SingleTaskGP

def make_beta(model: Model, X: Tensor) -> Dict[str, Any]:
    return {"beta": 0.1}

class UpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound (UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
        self,
        model: Model,
        beta: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        self.beta = self.beta.to(X)
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean
        view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
        mean = mean.view(view_shape)
        variance = posterior.variance.view(view_shape)
        delta = (self.beta.expand_as(mean) * variance).sqrt()
        if self.maximize:
            return mean + delta
        else:
            return -mean + delta
        

train_X = torch.rand(5, 1)
train_Y = torch.cos(train_X)

bounds = torch.stack([torch.zeros(1), torch.ones(1)])

model = SingleTaskGP(train_X, train_Y)

q = 1
beta = 10.0

qMS = qMultiStepLookahead(
model=model,
batch_sizes=[1, 1],
num_fantasies=[1, 1], # [4, 3],
valfunc_cls=[UpperConfidenceBound, UpperConfidenceBound, UpperConfidenceBound],
valfunc_argfacs=[make_beta, make_beta, make_beta],
)

q_prime = qMS.get_augmented_q_batch_size(q)
eval_X = torch.rand(q_prime, 1)

val = qMS(eval_X)

cands, vals = optimize_acqf(
acq_function=qMS,
bounds=bounds,
q=q_prime,
num_restarts=4,
raw_samples=16,
)
print('cands:', cands)