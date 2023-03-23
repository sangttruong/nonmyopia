import copy
import numpy as np
import torch
from torch import Tensor
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from botorch.acquisition import MCAcquisitionObjective
from botorch.acquisition.acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from botorch import settings
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    PosteriorTransform,
)

class qHEntropySearchTopK(MCAcquisitionFunction):
    def __init__(
            self,
            model: Model,
            dist_weight=1.0,
            dist_threshold=0.5,
            sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            posterior_transform: Optional[PosteriorTransform] = None,
            X_pending: Optional[Tensor] = None,
            **kwargs: Any,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        if sampler is None:
            sampler = SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)
        self.register_buffer("dist_weight", torch.as_tensor(dist_weight))
        self.register_buffer("dist_threshold", torch.as_tensor(dist_threshold))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qHEntropySearchTopKInner on X_fantasies.

        Args:
            X: batch_size x num_fantasies x num_actions x action_dim
        """
        K = X.shape[2]
        # Permute shape of X to work with self.model.posterior correctly
        X = torch.permute(X, [1, 0, 2, 3])

        posterior = self.model.posterior(X, posterior_transform=self.posterior_transform)
        samples = self.sampler(posterior)  # inner_MC x num_fantasies x batch_size x K x 1
        val = samples.squeeze(-1)  # inner_MC x num_fantasies x batch_size x K
        val = val.sum(dim=-1)  # inner_MC x num_fantasies x batch_size
        val = val.mean(dim=0)  # num_fantasies x batch_size

        if K >= 2:
            X_dist = torch.cdist(X.contiguous(), X.contiguous(), p=1.0)
            X_dist_triu = torch.triu(X_dist)  # num_fantasies x batch_size x K x K
            X_dist_triu[X_dist_triu > self.dist_threshold] = self.dist_threshold
            dist_reward = X_dist_triu.sum((-1, -2)) / (K * (K - 1) / 2.0)  # num_fantasies x batch_size
        else:
            dist_reward = 0.0  # num_fantasies x batch_size
        dist_reward = self.dist_weight * dist_reward

        q_hes = val + dist_reward
        q_hes = q_hes.squeeze()
        return q_hes


class qHEntropySearchMinMax(MCAcquisitionFunction):
    def __init__(
            self,
            model: Model,
            sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)
        self.sampler = sampler

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qHEntropySearchTopKInner on X_fantasies.

        Args:
            X: batch_size x num_fantasies x num_actions x action_dim
        """
        NUM_FANTASIES = X.shape[1]
        assert X.shape[2] == 2

        # Permute shape of X to work with self.model.posterior correctly
        X = torch.permute(X, [1, 0, 2, 3])

        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)  # inner_MC x num_fantasies x batch_size x K x 1
        val = samples.squeeze(-1)  # inner_MC x num_fantasies x batch_size x K
        val[:, :, :, 0] = -1 * val[:, :, :, 0]
        val = val.sum(dim=-1)  # inner_MC x num_fantasies x batch_size
        val = val.mean(dim=0)  # num_fantasies x batch_size
        q_hes = val

        return q_hes


class qHEntropySearchTwoVal(MCAcquisitionFunction):
    def __init__(
            self,
            model: Model,
            val_tuple=(-0.5, 0.5),
            sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)
        self.sampler = sampler
        self.register_buffer("val_tuple", torch.as_tensor(val_tuple))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qHEntropySearchTopKInner on X_fantasies.

        Args:
            X: batch_size x num_fantasies x num_actions x action_dim
        """
        NUM_FANTASIES = X.shape[1]
        assert X.shape[2] == 2
        K = X.shape[2]

        # Permute shape of X to work with self.model.posterior correctly
        X = torch.permute(X, [1, 0, 2, 3])

        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)  # inner_MC x num_fantasies x batch_size x K x 1
        val = samples.squeeze(-1)  # inner_MC x num_fantasies x batch_size x K
        val[:, :, :, 0] = -1 * torch.abs(val[:, :, :, 0] - self.val_tuple[0])
        val[:, :, :, 1] = -1 * torch.abs(val[:, :, :, 1] - self.val_tuple[1])
        val = val.sum(dim=-1)  # inner_MC x num_fantasies x batch_size
        val = val.mean(dim=0)  # num_fantasies x batch_size
        q_hes = val

        close = True
        if close:
            X_dist = torch.cdist(X.contiguous(), X.contiguous(), p=1.0)
            X_dist_triu = torch.triu(X_dist)  # num_fantasies x batch_size x K x K
            dist_reward = -1 * X_dist_triu.sum((-1, -2)) / (K * (K - 1) / 2.0)  # num_fantasies x batch_size
            dist_reward = 2 * dist_reward
            q_hes += dist_reward

        origin = True
        if origin:
            dist_origin_reward = -20 * torch.linalg.norm(X[:, :, 0, :], dim=-1)
            dist_origin_reward += -5 * torch.linalg.norm(X[:, :, 1, :], dim=-1)
            q_hes += dist_origin_reward

        return q_hes


class qHEntropySearchMVS(MCAcquisitionFunction):
    def __init__(
            self,
            model: Model,
            val_tuple=(-0.5, 0.5),
            sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)
        self.sampler = sampler
        self.register_buffer("val_tuple", torch.as_tensor(val_tuple))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qHEntropySearchTopKInner on X_fantasies.

        Args:
            X: batch_size x num_fantasies x num_actions x action_dim
        """
        NUM_FANTASIES = X.shape[1]
        K = X.shape[2]

        # Permute shape of X to work with self.model.posterior correctly
        X = torch.permute(X, [1, 0, 2, 3])

        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)  # inner_MC x num_fantasies x batch_size x K x 1
        val = samples.squeeze(-1)  # inner_MC x num_fantasies x batch_size x K
        for idx in range(K):
            #val[:, :, :, idx] = -1 * torch.abs(val[:, :, :, idx] - self.val_tuple[idx])**2
            val[:, :, :, idx] = -1 * torch.abs(val[:, :, :, idx] - self.val_tuple[idx])

        val = val.sum(dim=-1)  # inner_MC x num_fantasies x batch_size
        val = val.mean(dim=0)  # num_fantasies x batch_size
        q_hes = val

        close = False
        if close:
            X_dist = torch.cdist(X.contiguous(), X.contiguous(), p=1.0)
            X_dist_triu = torch.triu(X_dist)  # num_fantasies x batch_size x K x K
            dist_reward = -1 * X_dist_triu.sum((-1, -2)) / (K * (K - 1) / 2.0)  # num_fantasies x batch_size
            dist_reward = 1 * dist_reward
            q_hes += dist_reward

        origin = False
        if origin:
            dist_origin_reward = -20 * torch.linalg.norm(X[:, :, 0, :], dim=-1)
            dist_origin_reward += -5 * torch.linalg.norm(X[:, :, 1, :], dim=-1)
            q_hes += dist_origin_reward

        chain = True
        if chain:
            for idx in range(1, K):
                link_dist = X[:, :, idx, :] - X[:, :, idx-1, :]
                #link_dist_reward = -0.01 * torch.linalg.norm(link_dist, dim=-1)
                link_dist_reward = -0.1 * torch.linalg.norm(link_dist, dim=-1)
                q_hes += link_dist_reward


        return q_hes


class qHEntropySearchLevelSet(MCAcquisitionFunction):
    def __init__(
            self,
            model: Model,
            support_points,
            levelset_threshold,
            num_fantasies,
            batch_size,
            sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(num_samples=128, collapse_batch_dims=True)
        self.sampler = sampler
        self.register_buffer("support_points", torch.as_tensor(support_points))
        self.register_buffer("levelset_threshold", torch.as_tensor(levelset_threshold))
        self.support_points = self.support_points.repeat(num_fantasies, batch_size, 1, 1)
        # shape: num_fantasies x batch_size x num_actions x action_dim

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qHEntropySearchTopKInner on Actions.

        Args:
            support_points: num_actions x data_dim
            X: batch_size x num_fantasies x num_actions x action_dim
            where num_actions is the support size.
        """
        assert X.shape[3] == 1

        X = X.squeeze(-1).permute([1, 0, 2])  # num_fantasies x batch_size x num_actions
        posterior = self.model.posterior(self.support_points)
        samples = self.sampler(posterior)  # inner_MC x num_fantasies x batch_size x num_actions x 1
        val = samples.squeeze(-1)  # inner_MC x num_fantasies x batch_size x num_actions
        val = val.mean(dim=0)  # num_fantasies x batch_size x num_actions
        q_hes = ((val - self.levelset_threshold) * X).sum(dim=-1)  # num_fantasies x batch_size
        return q_hes


class qHEntropySearchMultiLevelSet(MCAcquisitionFunction):
    def __init__(
            self,
            model: Model,
            support_points,
            levelset_thresholds,
            num_fantasies,
            batch_size,
            sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(num_samples=128, collapse_batch_dims=True)
        self.sampler = sampler
        self.register_buffer("support_points", torch.as_tensor(support_points))
        self.register_buffer("levelset_thresholds", torch.as_tensor(levelset_thresholds))
        self.support_points = self.support_points.repeat(num_fantasies, batch_size, 1, 1)
        # shape: num_fantasies x batch_size x num_actions x action_dim

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qHEntropySearchTopKInner on Actions.

        Args:
            support_points: num_actions x data_dim
            X: batch_size x num_fantasies x num_actions x action_dim
            where num_actions is the support size.
        """
        assert X.shape[3] == len(self.levelset_thresholds)
        X = X.permute([1, 0, 2, 3])  # num_fantasies x batch_size x num_actions x num_levelset
        posterior = self.model.posterior(self.support_points)
        samples = self.sampler(posterior)  # inner_MC x num_fantasies x batch_size x num_actions x 1
        val = samples.squeeze(-1)  # inner_MC x num_fantasies x batch_size x num_actions
        val = val.mean(dim=0)  # num_fantasies x batch_size x num_actions
        q_hes = 0
        for i, threshold in enumerate(self.levelset_thresholds):
            q_hes += ((val - threshold) * X[:, :, :, i]).sum(dim=-1)  # num_fantasies x batch_size
        return q_hes


class qHEntropySearchExpf(MCAcquisitionFunction):
    def __init__(
            self,
            model: Model,
            dist_weight=1.0,
            dist_threshold=0.5,
            sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)
        self.sampler = sampler
        self.register_buffer("dist_weight", torch.as_tensor(dist_weight))
        self.register_buffer("dist_threshold", torch.as_tensor(dist_threshold))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qHEntropySearchTopKInner on X_fantasies.
        Args:
            X: batch_size x num_fantasies x num_actions x action_dim
        """
        NUM_FANTASIES = X.shape[1]
        K = X.shape[2]

        # Permute shape of X to work with self.model.posterior correctly
        X = torch.permute(X, [1, 0, 2, 3])

        # Draw samples from Normal distribution
        # --- Draw standard normal samples (of a certain shape)
        #std_normal = torch.normal(# TODO)
        # --- Transform, using X, to get to correct means/stds/weights (encoded in X)
        # TODO
        # --- Take function evals with self.model.posterior(samples)
        # TODO
        # --- Compute average of these function evals
        # TODO

        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)  # inner_MC x num_fantasies x batch_size x K x 1
        val = samples.squeeze(-1)  # inner_MC x num_fantasies x batch_size x K
        val = val.sum(dim=-1)  # inner_MC x num_fantasies x batch_size
        val = val.mean(dim=0)  # num_fantasies x batch_size

        X_dist = torch.cdist(X.contiguous(), X.contiguous(), p=1.0)
        X_dist_triu = torch.triu(X_dist)  # num_fantasies x batch_size x K x K
        X_dist_triu[X_dist_triu > self.dist_threshold] = self.dist_threshold
        dist_reward = X_dist_triu.sum((-1, -2)) / (K * (K - 1) / 2.0)  # num_fantasies x batch_size
        dist_reward = self.dist_weight * dist_reward

        q_hes = val + dist_reward
        q_hes = q_hes.squeeze()
        return q_hes


class qHEntropySearchPbest(MCAcquisitionFunction):
    def __init__(
            self,
            model: Model,
            num_restarts,
            num_fantasies,
            rand_samp,
            sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)
        self.sampler = sampler

        rand_samp = torch.tile(torch.tensor(rand_samp), (num_fantasies, num_restarts, 1, 1))
        self.posterior_rand_samp = self.model.posterior(rand_samp)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qHEntropySearchTopKInner on X_fantasies.

        Args:
            X: batch_size x num_fantasies x num_actions x action_dim
        """
        K = X.shape[2]

        # Permute shape of X to work with self.model.posterior correctly
        X = torch.permute(X, [1, 0, 2, 3])

        samples_rand_samp = self.sampler(self.posterior_rand_samp)
        maxes = torch.amax(samples_rand_samp, dim=(3, 4))

        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)   # out shape: inner_MC x num_fantasies x batch_size x K x 1

        val = -1 * (maxes - samples.squeeze())    # out shape: inner_MC x num_fantasies x batch_size
        val[val > -0.2] = -0.2
        val = val.mean(dim=0)               # out shape: num_fantasies x batch_size

        q_hes = val
        q_hes = q_hes.squeeze()
        return q_hes


class qHEntropySearchBestOfK(MCAcquisitionFunction):
    def __init__(
            self,
            model: Model,
            dist_weight=1.0,
            dist_threshold=0.5,
            sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)
        self.sampler = sampler
        self.register_buffer("dist_weight", torch.as_tensor(dist_weight))
        self.register_buffer("dist_threshold", torch.as_tensor(dist_threshold))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qHEntropySearchTopKInner on X_fantasies.

        Args:
            X: batch_size x num_fantasies x num_actions x action_dim
        """
        NUM_FANTASIES = X.shape[1]
        K = X.shape[2]

        # Permute shape of X to work with self.model.posterior correctly
        X = torch.permute(X, [1, 0, 2, 3])

        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)  # inner_MC x num_fantasies x batch_size x K x 1
        val = samples.squeeze(-1)  # inner_MC x num_fantasies x batch_size x K
        val = val.amax(dim=-1) # inner_MC x num_fantasies x batch_size
        val = val.mean(dim=0)  # num_fantasies x batch_size

        q_hes = val
        q_hes = q_hes.squeeze()
        return q_hes


class qHEntropySearch(MCAcquisitionFunction, OneShotAcquisitionFunction):
    r"""Batch H-Entropy Search using one-shot optimization.
    """

    def __init__(
            self,
            model: Model,
            config,
            sampler: Optional[MCSampler] = None,
            X_pending: Optional[Tensor] = None,
            inner_sampler: Optional[MCSampler] = None,
    ) -> None:
        r"""q-H-Entropy Search (one-shot optimization).

        Args:
            model: A fitted model. Must support fantasizing.
            num_fantasies: number of samples for outer expectation
            K: K in top-K task, the optimization dim for inner problem is K x d
            sampler: The sampler used to sample fantasy observations. Optional
                if `num_fantasies` is specified.
            inner_sampler: The sampler used for inner sampling. Ignored if the
                objective is `None` or a ScalarizedObjective.
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
        if inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(
                num_samples=16,  # 128, or 64
                resample=False, collapse_batch_dims=True
            )
        self.sampler = sampler
        self.inner_sampler = inner_sampler
        self.num_fantasies = config.num_outer_mc
        self.num_actions = config.num_action
        self.design_dim = config.num_dim_design
        self.action_dim = config.num_dim_action
        self.config = config
        self.set_initialize_func()
        self.set_X_pending(X_pending)

    def forward(self, X_actual: Tensor, X_fantasies: Tensor) -> Tensor:
        r"""Evaluate qKnowledgeGradient on the candidate set `X`.

        Args:
            X_actual: b x q x d_design
            X_fantasies: b x num_actions x d_action

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KG value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X_actual[b]`, and `X_fantasies[b, : ]` must be
                maximized at fixed `X_actual[b]`.
        """
        # X_actual, X_fantasies = _split_fantasy_points(X=X,
        #                                               num_actions=self.num_actions,
        #                                               design_dim=self.design_dim,
        #                                               action_dim=self.action_dim)

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X_actual, sampler=self.sampler, observation_noise=True
        )

        # get the value function
        if self.config.app == 'topk':
            self.value_function_cls = qHEntropySearchTopK
            value_function = self.value_function_cls(
                model=fantasy_model,
                sampler=self.inner_sampler,
                dist_weight=self.config.dist_weight,
                dist_threshold=self.config.dist_threshold,
            )
        elif self.config.app == 'minmax':
            self.value_function_cls = qHEntropySearchMinMax
            value_function = self.value_function_cls(
                model=fantasy_model,
                sampler=self.inner_sampler,
            )
        elif self.config.app == 'twoval':
            self.value_function_cls = qHEntropySearchTwoVal
            value_function = self.value_function_cls(
                model=fantasy_model,
                sampler=self.inner_sampler,
                val_tuple=self.config.val_tuple,
            )
        elif self.config.app == 'mvs':
            self.value_function_cls = qHEntropySearchMVS
            value_function = self.value_function_cls(
                model=fantasy_model,
                val_tuple=self.config.val_tuple,
                sampler=self.inner_sampler,
            )
        elif self.config.app == 'levelset':
            self.value_function_cls = qHEntropySearchLevelSet
            value_function = self.value_function_cls(
                model=fantasy_model,
                sampler=self.inner_sampler,
                support_points=self.config.support_points,
                levelset_threshold=self.config.levelset_threshold,
                num_fantasies=self.config.num_outer_mc,
                batch_size=self.config.batch_size,
            )
        elif self.config.app == 'multilevelset':
            self.value_function_cls = qHEntropySearchMultiLevelSet
            value_function = self.value_function_cls(
                model=fantasy_model,
                sampler=self.inner_sampler,
                support_points=self.config.support_points,
                levelset_thresholds=self.config.levelset_thresholds,
                num_fantasies=self.config.num_outer_mc,
                batch_size=self.config.batch_size,
            )
        elif self.config.app == 'pbest':
            self.value_function_cls = qHEntropySearchPbest
            value_function = self.value_function_cls(
                model=fantasy_model,
                num_restarts=self.config.num_restarts,
                num_fantasies=self.config.num_outer_mc,
                rand_samp=self.config.rand_samp,
                sampler=self.inner_sampler,
            )
        elif self.config.app == 'bestofk':
            self.value_function_cls = qHEntropySearchBestOfK
            value_function = self.value_function_cls(
                model=fantasy_model,
                sampler=self.inner_sampler,
                dist_weight=self.config.dist_weight,
                dist_threshold=self.config.dist_threshold,
            )

        # make sure to propagate gradients to the fantasy model train inputs
        with settings.propagate_grads(True):
            values = value_function(X=X_fantasies)  # num_fantasies x batch_size
        # return average over the fantasy samples
        return values.mean(dim=0)

    def get_augmented_q_batch_size(self, q: int) -> int:
        return q + self.action_dim

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        X_actual, X_fantasies = _split_fantasy_points(X=X_full,
                                                      num_actions=self.num_actions,
                                                      design_dim=self.design_dim,
                                                      action_dim=self.action_dim)
        return X_actual

    def set_initialize_func(self):
        """Set self.initialize_design_action_tensors."""
        if self.config.app == 'mvs':
            self.initialize_design_action_tensors = self.initialize_tensors_mvs
        else:
            self.initialize_design_action_tensors = self.initialize_tensors

    def initialize_tensors(self, data):
        """Initialize and return X_design, X_action tensors."""
        config = self.config # for brevity
        bounds_diff_design = config.bounds_design[1] - config.bounds_design[0]
        bounds_diff_action = config.bounds_action[1] - config.bounds_action[0]

        X_design = config.bounds_design[0] + bounds_diff_design * torch.rand(
            config.num_restarts,
            config.num_candidates,
            config.num_dim_design,
        )
        X_design.requires_grad_(True)

        X_action = config.bounds_action[0] + bounds_diff_action * torch.rand(
            config.num_restarts,
            config.num_outer_mc,
            config.num_action,
            config.num_dim_action,
        )

        # Initialize actions consistently across fantasies
        X_action[:, 1:, :, :] = X_action[:, 0:1, :, :]

        X_action.requires_grad_(True)

        return X_design, X_action

    def initialize_tensors_topk(self, data):
        """[WIP] Initialize and return X_design, X_action tensors for topk."""
        X_design, X_action = self.initialize_tensors(data)

        # Initialize actions to topk diverse data points
        config = self.config # for brevity
        data_y = copy.deepcopy(np.array(data.y).reshape(-1))
        data_x = copy.deepcopy(np.array(data.x).reshape(-1, config.num_dim_design))
        for i in range(config.num_action):
            if len(data_y) > 0:
                topk_idx = data_y.argmax()
                topk_x = data_x[topk_idx]
                dists = np.linalg.norm(data_x - topk_x, axis=1, ord=1)
                del_idx = np.where(dists < config.dist_threshold)[0]
                data_x = np.delete(data_x, del_idx, axis=0)
                data_y = np.delete(data_y, del_idx, axis=0)
                print(f'topk_x {i} = {topk_x}')
                with torch.no_grad():
                    X_action[:, :, i, :] = torch.tensor(topk_x)
            else:
                pass

        return X_design, X_action

    def initialize_tensors_mvs(self, data):
        """[WIP] Initialize and return X_design, X_action tensors for topk."""
        X_design, X_action = self.initialize_tensors(data)

        # Initialize actions to topk diverse data points
        config = self.config # for brevity
        data_y = copy.deepcopy(np.array(data.y).reshape(-1))
        data_x = copy.deepcopy(np.array(data.x).reshape(-1, config.num_dim_design))

        if len(data_y) > 0:
            argmax_x = data_x[data_y.argmax()]
            argmin_x = data_x[data_y.argmin()]
            init_arr = np.linspace(argmax_x, argmin_x, config.num_action)

            for i, init_x in enumerate(init_arr):
                with torch.no_grad():
                    X_action[:, :, i, :] = torch.tensor(init_x)

        return X_design, X_action


def _split_fantasy_points(X: Tensor, num_actions: int, design_dim: int, action_dim: int):
    assert len(X.shape) == 2
    batch_size = X.size(0)
    split_sizes = [X.size(1) - num_actions * action_dim, num_actions * action_dim]
    X_actual, X_fantasies = torch.split(X, split_sizes, dim=1)
    X_actual = X_actual.reshape(batch_size, -1, design_dim)
    X_fantasies = X_fantasies.reshape(batch_size, num_actions, action_dim)
    return X_actual, X_fantasies
