#!/usr/bin/env python
"""Batched H-entropy Search acquisition function. Some supported losses 
functions are: TODO: fill-out file description
(1) Top-K with diversity
(2) MinMax
(3) TwoVal
(4) MVS
(5) LevelSet
(6) MultiLevelSet
(7) Expf
(8) Pbest
(9) BestOfK
"""

from torchtyping import TensorType, patch_typeguard
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.models.model import Model
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from typing import Optional

__author__ = ""
__copyright__ = "Copyright 2022, Stanford University"

import torch
import copy
import numpy as np

patch_typeguard()
torch.autograd.set_detect_anomaly(True)


class HEntropySearchTopK(MCAcquisitionFunction):
    r"""Evaluate a batch of H-entropy acquisition function for top-K with diversity."""

    def __init__(
        self,
        config,
        model: Model,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        r"""
        Note that the model object `self.model` here is the result of fantasization
        on the intial model. In other words, it is a GP of p (f | D1)
        = p( f | D U {batch_xs, Y}), where Y is sample from posterior p( f | D ).
        This GP represents the inner distribution that we need to compute the expectation
        over.
        """

        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(
                num_samples=512, collapse_batch_dims=True)
        self.sampler = sampler
        self.register_buffer(
            "dist_weight", torch.as_tensor(config.dist_weight))
        self.register_buffer(
            "dist_threshold", torch.as_tensor(config.dist_threshold))
        self.config = config

    # @typechecked
    def forward(self, po) -> TensorType["n_restarts"]:
        """See equation 7 in the H-entropy paper.
        For a given function f and action a, the loss function is defined as
            l(f, a) = \sum_i f(a_i) + \sum_{1 \leq i \leq j \leq k} d(a_i, a_j)
        Then the corresponding acquisition function is defined as
            E_{p(y|x,D)} E_{f|D_1} [ l(f, a) ]
        Value of the acquisition function is approximated using Monte Carlo.
        """
        c = self.config
        K = c.n_actions
        po[-1] = po[-1].reshape(1, 2)

        p_yi_xiDi = self.model.posterior(po[c.lookahead_steps])
        batch_yis = self.sampler(p_yi_xiDi)
        # >> Tensor[ n_inner_samples, *[n_samples]*i, n_restarts, 1, 1]

        # sum over K dimension
        # batch_yis = batch_yis.sum(dim=-2, keep_dim=True)

        batch_yis = batch_yis.mean(dim=0)
        # >> Tensor[*[n_samples]*i, n_restarts, 1, 1]

        # compute pairwise-distance d(a_i, a_j) for the diversity
        if K >= 2:
            batch_as_dist = torch.cdist(
                po[c.lookahead_steps].contiguous(),
                po[c.lookahead_steps].contiguous(),
                p=1.0,
            )
            # n_samples x n_restarts x K x K
            batch_as_dist_triu = torch.triu(batch_as_dist)
            batch_as_dist_triu[
                batch_as_dist_triu > self.dist_threshold
            ] = self.dist_threshold
            dist_reward = batch_as_dist_triu.sum((-1, -2)) / (
                K * (K - 1) / 2.0
            )  # n_samples x n_restarts
        else:
            dist_reward = 0.0  # n_samples x n_restarts
        dist_reward = self.dist_weight * dist_reward

        total_cost = 0
        if c.r:
            threshold1 = torch.nn.Threshold(-c.r, 100)
            threshold2 = torch.nn.Threshold(0, 0)
            for i in range(c.lookahead_steps):
                distance = ((po[i] - po[i - 1]) **
                            2).sum((-1, -2), keepdim=True)
                cost = threshold2(threshold1(-distance))
                total_cost = total_cost + cost

        # sum over samples from posterior predictive
        self.result = batch_yis - total_cost + dist_reward

        # compute the advantage
        # if c.baseline is not None: result = result - c.baseline
        self.result = self.result.squeeze()
        avg_result = self.result
        while len(avg_result.shape) > 1:
            avg_result = avg_result.mean(0)
        return avg_result

    # @typechecked
    def directly_parameterize_output_topk(self, data):
        """[WIP] init and return batch_x0s, batch_a1s tensors for topk."""
        mc_params = self.directly_parameterize_output(data)

        batch_x0s = mc_params[0]
        batch_a1s = mc_params[self.config.lookahead_steps]

        # init actions to topk diverse data points
        config = self.config  # for brevity
        data_y = copy.deepcopy(np.array(data.y).reshape(-1))
        data_x = copy.deepcopy(
            np.array(data.x.cpu()).reshape(-1, config.n_dim_design))
        for i in range(config.n_actions):
            if len(data_y) > 0:
                topk_idx = data_y.argmax()
                topk_x = data_x[topk_idx]
                dists = np.linalg.norm(data_x - topk_x, axis=1, ord=1)
                del_idx = np.where(dists < config.dist_threshold)[0]
                data_x = np.delete(data_x, del_idx, axis=0)
                data_y = np.delete(data_y, del_idx, axis=0)
                print(f"topk_x {i} = {topk_x}")
                with torch.no_grad():
                    batch_a1s[:, :, i, :] = torch.tensor(topk_x)
            else:
                pass

        return batch_x0s, batch_a1s.to(self.config.device)


class HEntropySearchMinMax(MCAcquisitionFunction):
    def __init__(
        self,
        config,
        model: Model,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(
                num_samples=512, collapse_batch_dims=True)
        self.sampler = sampler
        self.config = config

    def forward(self, batch_as: torch.Tensor) -> torch.Tensor:
        r"""Evaluate HEntropySearchTopKInner on batch_as.

        Args:
            batch_as: n_restarts x n_samples x n_actions x action_dim
        """
        assert batch_as.shape[2] == 2

        # Permute shape of batch_as to work with self.model.posterior correctly
        batch_as = torch.permute(batch_as, [1, 0, 2, 3])

        posterior = self.model.posterior(batch_as)
        # n_fs x n_samples x n_restarts x K x 1
        samples = self.sampler(posterior)
        val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
        val[:, :, :, 0] = -1 * val[:, :, :, 0]
        val = val.sum(dim=-1)  # n_fs x n_samples x n_restarts
        val = val.mean(dim=0)  # n_samples x n_restarts
        q_hes = val

        return q_hes.mean(dim=0)


class HEntropySearchTwoVal(MCAcquisitionFunction):
    def __init__(
        self,
        config,
        model: Model,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(
                num_samples=512, collapse_batch_dims=True)
        self.sampler = sampler
        self.register_buffer("val_tuple", torch.as_tensor(config.val_tuple))
        self.config = config

    def forward(self, batch_as: torch.Tensor) -> torch.Tensor:
        r"""Evaluate HEntropySearchTopKInner on batch_as.

        Args:
            batch_as: n_restarts x n_samples x n_actions x action_dim
        """
        K = self.config.n_actions
        assert K == 2

        # Permute shape of batch_as to work with self.model.posterior correctly
        batch_as = torch.permute(batch_as, [1, 0, 2, 3])

        posterior = self.model.posterior(batch_as)
        # n_fs x n_samples x n_restarts x K x 1
        samples = self.sampler(posterior)
        val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
        val[:, :, :, 0] = -1 * torch.abs(val[:, :, :, 0] - self.val_tuple[0])
        val[:, :, :, 1] = -1 * torch.abs(val[:, :, :, 1] - self.val_tuple[1])
        val = val.sum(dim=-1)  # n_fs x n_samples x n_restarts
        val = val.mean(dim=0)  # n_samples x n_restarts
        q_hes = val

        close = True
        if close:
            batch_as_dist = torch.cdist(
                batch_as.contiguous(), batch_as.contiguous(), p=1.0
            )
            # n_samples x n_restarts x K x K
            batch_as_dist_triu = torch.triu(batch_as_dist)
            # n_samples x n_restarts
            dist_reward = -1 * \
                batch_as_dist_triu.sum((-1, -2)) / (K * (K - 1) / 2.0)
            dist_reward = 2 * dist_reward
            q_hes += dist_reward

        origin = True
        if origin:
            dist_origin_reward = -20 * \
                torch.linalg.norm(batch_as[:, :, 0, :], dim=-1)
            dist_origin_reward += -5 * \
                torch.linalg.norm(batch_as[:, :, 1, :], dim=-1)
            q_hes += dist_origin_reward

        return q_hes.mean(dim=0)


class HEntropySearchMVS(MCAcquisitionFunction):
    def __init__(
        self,
        config,
        model: Model,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(
                num_samples=512, collapse_batch_dims=True)
        self.sampler = sampler
        self.register_buffer("val_tuple", torch.as_tensor(config.val_tuple))
        self.config = config

    def forward(self, batch_as: torch.Tensor) -> torch.Tensor:
        r"""Evaluate HEntropySearchTopKInner on batch_as.

        Args:
            batch_as: n_restarts x n_samples x n_actions x action_dim
        """

        K = self.config.n_actions

        # Permute shape of batch_as to work with self.model.posterior correctly
        batch_as = torch.permute(batch_as, [1, 0, 2, 3])

        posterior = self.model.posterior(batch_as)
        # n_fs x n_samples x n_restarts x K x 1
        samples = self.sampler(posterior)
        val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
        for idx in range(K):
            # val[:, :, :, idx] = -1 * torch.abs(val[:, :, :, idx] - self.val_tuple[idx])**2
            val[:, :, :, idx] = -1 * \
                torch.abs(val[:, :, :, idx] - self.val_tuple[idx])

        val = val.sum(dim=-1)  # n_fs x n_samples x n_restarts
        val = val.mean(dim=0)  # n_samples x n_restarts
        q_hes = val

        close = False
        if close:
            batch_as_dist = torch.cdist(
                batch_as.contiguous(), batch_as.contiguous(), p=1.0
            )
            # n_samples x n_restarts x K x K
            batch_as_dist_triu = torch.triu(batch_as_dist)
            # n_samples x n_restarts
            dist_reward = -1 * \
                batch_as_dist_triu.sum((-1, -2)) / (K * (K - 1) / 2.0)
            dist_reward = 1 * dist_reward
            q_hes += dist_reward

        origin = False
        if origin:
            dist_origin_reward = -20 * \
                torch.linalg.norm(batch_as[:, :, 0, :], dim=-1)
            dist_origin_reward += -5 * \
                torch.linalg.norm(batch_as[:, :, 1, :], dim=-1)
            q_hes += dist_origin_reward

        chain = True
        if chain:
            for idx in range(1, K):
                link_dist = batch_as[:, :, idx, :] - batch_as[:, :, idx - 1, :]
                # link_dist_reward = -0.01 * torch.linalg.norm(link_dist, dim=-1)
                link_dist_reward = -0.1 * torch.linalg.norm(link_dist, dim=-1)
                q_hes += link_dist_reward

        return q_hes.mean(dim=0)

    # @typechecked
    def directly_parameterize_output_mvs(self, data):
        """[WIP] init and return batch_x0s, batch_a1s tensors for topk."""
        batch_x0s, batch_a1s = self.directly_parameterize_output(data)

        # init actions to topk diverse data points
        c = self.config  # for brevity
        data_y = copy.deepcopy(np.array(data.y).reshape(-1))
        data_x = copy.deepcopy(np.array(data.x).reshape(-1, c.n_dim_design))

        if len(data_y) > 0:
            argmax_x = data_x[data_y.argmax()]
            argmin_x = data_x[data_y.argmin()]
            init_arr = np.linspace(argmax_x, argmin_x, c.n_action)

            for i, init_x in enumerate(init_arr):
                with torch.no_grad():
                    batch_a1s[:, :, i, :] = torch.tensor(init_x)

        return batch_x0s, batch_a1s


class HEntropySearchLevelSet(MCAcquisitionFunction):
    def __init__(
        self,
        config,
        model: Model,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(
                num_samples=128, collapse_batch_dims=True)
        self.sampler = sampler
        self.register_buffer(
            "support_points", torch.as_tensor(config.support_points))
        self.register_buffer(
            "levelset_threshold", torch.as_tensor(config.levelset_threshold)
        )
        self.support_points = self.support_points.repeat(
            config.n_samples, config.n_restarts, 1, 1
        )
        self.config = config
        # shape: n_samples x n_restarts x n_actions x action_dim

    def forward(self, batch_as: torch.Tensor) -> torch.Tensor:
        r"""Evaluate HEntropySearchTopKInner on Actions.

        Args:
            support_points: n_actions x data_dim
            batch_as: n_restarts x n_samples x n_actions x action_dim
            where n_actions is the support size.
        """
        assert batch_as.shape[3] == 1

        # n_samples x n_restarts x n_actions
        batch_as = batch_as.squeeze(-1).permute([1, 0, 2])
        posterior = self.model.posterior(self.support_points)
        # n_fs x n_samples x n_restarts x n_actions x 1
        samples = self.sampler(posterior)
        val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x n_actions
        val = val.mean(dim=0)  # n_samples x n_restarts x n_actions
        q_hes = ((val - self.levelset_threshold) * batch_as).sum(
            dim=-1
        )  # n_samples x n_restarts
        return q_hes.mean(dim=0)


class HEntropySearchMultiLevelSet(MCAcquisitionFunction):
    def __init__(
        self,
        config,
        model: Model,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(
                num_samples=128, collapse_batch_dims=True)
        self.sampler = sampler
        self.register_buffer(
            "support_points", torch.as_tensor(config.support_points))
        self.register_buffer(
            "levelset_thresholds", torch.as_tensor(config.levelset_thresholds)
        )
        self.support_points = self.support_points.repeat(
            config.n_samples, config.n_restarts, 1, 1
        )
        self.config = config
        # shape: n_samples x n_restarts x n_actions x action_dim

    def forward(self, batch_as: torch.Tensor) -> torch.Tensor:
        r"""Evaluate HEntropySearchTopKInner on Actions.

        Args:
            support_points: n_actions x data_dim
            batch_as: n_restarts x n_samples x n_actions x action_dim
            where n_actions is the support size.
        """
        assert batch_as.shape[3] == len(self.levelset_thresholds)
        # n_samples x n_restarts x n_actions x n_levelset
        batch_as = batch_as.permute([1, 0, 2, 3])
        posterior = self.model.posterior(self.support_points)
        # n_fs x n_samples x n_restarts x n_actions x 1
        samples = self.sampler(posterior)
        val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x n_actions
        val = val.mean(dim=0)  # n_samples x n_restarts x n_actions
        q_hes = 0
        for i, threshold in enumerate(self.levelset_thresholds):
            # n_samples x n_restarts
            q_hes += ((val - threshold) * batch_as[:, :, :, i]).sum(dim=-1)
        return q_hes.mean(dim=0)


class HEntropySearchExpf(MCAcquisitionFunction):
    def __init__(
        self,
        config,
        model: Model,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(
                num_samples=512, collapse_batch_dims=True)
        self.sampler = sampler
        self.register_buffer(
            "dist_weight", torch.as_tensor(config.dist_weight))
        self.register_buffer(
            "dist_threshold", torch.as_tensor(config.dist_threshold))
        self.config = config

    def forward(self, batch_as: torch.Tensor) -> torch.Tensor:
        r"""Evaluate HEntropySearchTopKInner on batch_as.
        Args:
            batch_as: n_restarts x n_samples x n_actions x action_dim
        """

        K = self.config.n_actions

        # Permute shape of batch_as to work with self.model.posterior correctly
        batch_as = torch.permute(batch_as, [1, 0, 2, 3])

        # Draw samples from Normal distribution
        # --- Draw standard normal samples (of a certain shape)
        # std_normal = torch.normal(# TODO)
        # --- Transform, using batch_as, to get to correct means/stds/weights (encoded in batch_as)
        # TODO
        # --- Take function evals with self.model.posterior(samples)
        # TODO
        # --- Compute average of these function evals
        # TODO

        posterior = self.model.posterior(batch_as)
        # n_fs x n_samples x n_restarts x K x 1
        samples = self.sampler(posterior)
        val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
        val = val.sum(dim=-1)  # n_fs x n_samples x n_restarts
        val = val.mean(dim=0)  # n_samples x n_restarts

        batch_as_dist = torch.cdist(
            batch_as.contiguous(), batch_as.contiguous(), p=1.0)
        # n_samples x n_restarts x K x K
        batch_as_dist_triu = torch.triu(batch_as_dist)
        batch_as_dist_triu[
            batch_as_dist_triu > self.dist_threshold
        ] = self.dist_threshold
        dist_reward = batch_as_dist_triu.sum((-1, -2)) / (
            K * (K - 1) / 2.0
        )  # n_samples x n_restarts
        dist_reward = self.dist_weight * dist_reward

        q_hes = val + dist_reward
        q_hes = q_hes.squeeze()
        return q_hes.mean(dim=0)


class HEntropySearchPbest(MCAcquisitionFunction):
    def __init__(
        self,
        config,
        model: Model,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(
                num_samples=512, collapse_batch_dims=True)
        self.sampler = sampler

        rand_samp = torch.tile(
            torch.tensor(rand_samp), (config.n_samples,
                                      config.n_restarts, 1, 1)
        )
        self.posterior_rand_samp = self.model.posterior(rand_samp)
        self.config = config

    def forward(self, batch_as: torch.Tensor) -> torch.Tensor:
        r"""Evaluate HEntropySearchTopKInner on batch_as.

        Args:
            batch_as: n_restarts x n_samples x n_actions x action_dim
        """

        K = self.config.n_actions

        # Permute shape of batch_as to work with self.model.posterior correctly
        batch_as = torch.permute(batch_as, [1, 0, 2, 3])

        samples_rand_samp = self.sampler(self.posterior_rand_samp)
        maxes = torch.amax(samples_rand_samp, dim=(3, 4))

        posterior = self.model.posterior(batch_as)
        # out shape: n_fs x n_samples x n_restarts x K x 1
        samples = self.sampler(posterior)

        # out shape: n_fs x n_samples x n_restarts
        val = -1 * (maxes - samples.squeeze())
        val[val > -0.2] = -0.2
        val = val.mean(dim=0)  # out shape: n_samples x n_restarts

        q_hes = val
        q_hes = q_hes.squeeze()
        return q_hes.mean(dim=0)


class HEntropySearchBestOfK(MCAcquisitionFunction):
    def __init__(
        self,
        config,
        model: Model,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(
                num_samples=512, collapse_batch_dims=True)
        self.sampler = sampler
        self.register_buffer(
            "dist_weight", torch.as_tensor(config.dist_weight))
        self.register_buffer(
            "dist_threshold", torch.as_tensor(config.dist_threshold))
        self.config = config

    def forward(self, batch_as: torch.Tensor) -> torch.Tensor:
        r"""Evaluate HEntropySearchTopKInner on batch_as.

        Args:
            batch_as: n_restarts x n_samples x n_actions x action_dim
        """

        K = self.config.n_actions

        # Permute shape of batch_as to work with self.model.posterior correctly
        batch_as = torch.permute(batch_as, [1, 0, 2, 3])

        posterior = self.model.posterior(batch_as)
        # n_fs x n_samples x n_restarts x K x 1
        samples = self.sampler(posterior)
        val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
        val = val.amax(dim=-1)  # n_fs x n_samples x n_restarts
        val = val.mean(dim=0)  # n_samples x n_restarts

        q_hes = val
        q_hes = q_hes.squeeze()
        return q_hes.mean(dim=0)
