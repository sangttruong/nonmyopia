import torch
import numpy as np
from _7_utils import kern_exp_quad_noard, sample_mvn, gp_post, unif_random_sample_domain


class SynGP:
    """Synthetic functions defined by draws from a Gaussian process."""

    def __init__(self, dim, seed=8, noise_std=0.0):
        self.bounds = torch.tensor([[-1, 1]] * dim).T
        self.dim = dim
        self.seed = seed
        self.noise_std = noise_std
        self.n_obs = 10
        self.hypers = {"ls": 0.25, "alpha": 10.0, "sigma": 1e-2, "n_dimx": dim}
        # self.hypers = {"ls": 0.25, "alpha": 1.0, "sigma": 1e-2, "n_dimx": dim}
        self.domain_samples = None
        self.prior_samples = None
        self.dtype = torch.float64
        self.device = torch.device("cpu")

    def initialize(self):
        """Initialize synthetic function."""
        self.set_random_seed()
        self.set_kernel()
        self.draw_domain_samples()
        self.draw_prior_samples()

    def set_random_seed(self):
        """Set random seed."""
        np.random.seed(self.seed)

    def set_kernel(self):
        """Set self.kernel function."""

        def kernel(xlist1, xlist2, ls, alpha):
            return kern_exp_quad_noard(xlist1, xlist2, ls, alpha)

        self.kernel = kernel

    def draw_domain_samples(self):
        """Draw uniform random samples from self.domain."""
        domain_samples = unif_random_sample_domain(self.bounds.T, self.n_obs)
        self.domain_samples = np.array(domain_samples).reshape(self.n_obs, -1)

    def draw_prior_samples(self):
        """Draw a prior function and evaluate it at self.domain_samples."""
        domain_samples = self.domain_samples
        prior_mean = np.zeros(domain_samples.shape[0])
        prior_cov = self.kernel(
            domain_samples, domain_samples, self.hypers["ls"], self.hypers["alpha"]
        )
        prior_samples = sample_mvn(prior_mean, prior_cov, 1)
        self.prior_samples = prior_samples.reshape(self.n_obs, -1)

    def __call__(self, test_x):
        """
        Call synthetic function on test_x, and return the posterior mean given by
        self.get_post_mean method.
        """
        if self.domain_samples is None or self.prior_samples is None:
            self.initialize()

        test_x = self.process_function_input(test_x)
        post_mean = self.get_post_mean(test_x)
        test_y = self.process_function_output(post_mean)

        return test_y

    def get_post_mean(self, test_x):
        """
        Return mean of model posterior (given self.domain_samples, self.prior_samples)

        at the test_x inputs.
        """
        post_mean, _ = gp_post(
            self.domain_samples,
            self.prior_samples,
            test_x,
            self.hypers["ls"],
            self.hypers["alpha"],
            self.hypers["sigma"],
            self.kernel,
        )

        return post_mean

    def process_function_input(self, test_x):
        """Process and possibly reshape inputs to the synthetic function."""
        self.device = test_x.device
        test_x = test_x.cpu().detach().numpy()
        if len(test_x.shape) == 1:
            test_x = test_x.reshape(1, -1)
            self.input_mode = "single"
        elif len(test_x.shape) == 0:
            assert self.hypers["n_dimx"] == 1
            test_x = test_x.reshape(1, -1)
            self.input_mode = "single"
        else:
            self.input_mode = "batch"

        return test_x

    def process_function_output(self, func_output):
        """Process and possibly reshape output of the synthetic function."""
        if self.input_mode == "single":
            func_output = func_output[0][0]
        elif self.input_mode == "batch":
            func_output = func_output.reshape(-1, 1)

        return torch.tensor(func_output, dtype=self.dtype, device=self.device)

    def to(self, dtype, device):
        self.dtype = dtype
        self.device = device
        return self
