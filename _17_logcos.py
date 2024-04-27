import torch
import numpy as np
from _7_utils import kern_exp_quad_noard, sample_mvn, gp_post, unif_random_sample_domain


class LogCos:
    """Synthetic functions defined by draws from a Gaussian process."""

    def __init__(self, dim, seed=8, noise_std=0.0):
        self.bounds = torch.tensor([[-1, 1]] * dim).T
        assert dim == 2, "Only 2D functions are supported."
        self.dim = dim
        self.seed = seed
        self.noise_std = noise_std
        self.dtype = torch.float64
        self.device = torch.device("cpu")

    def __call__(self, x):
        """
        Call synthetic function on test_x, and return the posterior mean given by
        self.get_post_mean method.
        """
        val = torch.log(torch.pow(x[..., 0], 2) +
                        torch.pow(x[..., 0], x[..., 1])) - \
            torch.cos(torch.pow(x[..., 1], x[..., 0]))

        val += self.noise_std * torch.randn_like(val)
        val = val.to(self.dtype).to(self.device)
        return val

    def to(self, dtype, device):
        self.dtype = dtype
        self.device = device
        return self
