#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Sampler modules producing N(0,1) samples, to be used with MC-evaluated
acquisition functions and Gaussian posteriors.
"""

from __future__ import annotations

import torch
from botorch.exceptions import UnsupportedError
from botorch.posteriors import Posterior
from botorch.exceptions import UnsupportedError
from botorch.posteriors import Posterior
from botorch.sampling.normal import NormalMCSampler
from botorch.utils.sampling import draw_sobol_normal_samples
from torch.quasirandom import SobolEngine


class DesireSobolQMCNormalSampler(NormalMCSampler):
    r"""Sampler for quasi-MC N(0,1) base samples using Sobol sequences.
    Example:
        >>> sampler = SobolQMCNormalSampler(1024, seed=1234)
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def _construct_base_samples(self, posterior: Posterior) -> None:
        r"""Generate quasi-random Normal base samples (if necessary).
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
            base_collapsed_shape = target_shape[len(self.sample_shape) :]
            output_dim = base_collapsed_shape.numel()
            if output_dim > SobolEngine.MAXDIM:
                raise UnsupportedError(
                    "SobolQMCSampler only supports dimensions "
                    f"`q * o <= {SobolEngine.MAXDIM}`. Requested: {output_dim}"
                )
            base_samples = draw_sobol_normal_samples(
                d=output_dim,
                n=self.sample_shape.numel(),
                device=posterior.device,
                dtype=posterior.dtype,
                seed=self.seed,
            )
            base_samples = base_samples.view(target_shape)

            # Rescale base_samples
            std = 1e-2
            base_samples = base_samples * std

            self.register_buffer("base_samples", base_samples)
        self.to(device=posterior.device, dtype=posterior.dtype)
