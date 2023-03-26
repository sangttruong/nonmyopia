import torch
import dill as pickle
import numpy as np
from utils.gp_utils import kern_exp_quad_noard, sample_mvn, gp_post
from utils.domain_utils import unif_random_sample_domain


def make(parms):
    # TODO: for other synthetic function like Alpine, import from BoTorch
    
    if parms.env_name == "SynGP":
        sf = SynGP(
            seed=parms.seed_synthfunc,
            x_dim=parms.x_dim,
            n_obs=parms.n_obs, 
            dtype=parms.torch_dtype
        )

    elif parms.env_name == "chemical":
        with open("examples/semisynthetic.pt", "rb") as file_handle:
            sf = pickle.load(file_handle)
    
    else:
        raise NotImplemented

    return sf


class SynGP:
    """Synthetic functions defined by draws from a Gaussian process."""

    def __init__(self, seed=12, x_dim=2, n_obs=50, dtype=None):

        self.learn_hypers = False
        self.bounds = [-1, 1]
        self.domain = [self.bounds] * x_dim
        self.n_obs = 50


        self.seed = seed
        self.hypers = {"ls": 0.1, "alpha": 2.0, "sigma": 1e-2, "n_dimx": x_dim}
        self.n_obs = n_obs
        self.dtype = dtype
        self.initialize()

    # @np.vectorize
    def vec(self, x, y):
        """Return f on input = (x, y)."""
        return np.reshape(self(np.stack((x.flatten(), y.flatten()), axis=-1)), x.shape)

    def func(self, x_list):
        """Synthetic function with torch tensor input/output."""
        x_list = [xi.cpu().detach().numpy().tolist() for xi in x_list]
        x_list = np.array(x_list) # .reshape(-1, args.n_dim)
        y_list = [self(x) for x in x_list]
        y_list = y_list[0] if len(y_list) == 1 else y_list
        y_tensor = torch.tensor(np.array(y_list).reshape(-1, 1), dtype=self.dtype)
        return y_tensor
    
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
        domain_samples = unif_random_sample_domain(self.domain, self.n_obs)
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
            self.kernel
        )
        return post_mean

    def process_function_input(self, test_x):
        """Process and possibly reshape inputs to the synthetic function."""
        test_x = np.array(test_x)
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
            func_output = func_output.reshape(-1)

        return func_output

    def set_ground_truth_GP_hyperparameters(self, WM):
        print(f"config.learn_hypers={self.learn_hypers}, using hypers from config.hypers")
        # NOTE: GPyTorch outputscale should be set to the SynthFunc alpha squared

        WM.covar_module.base_kernel.lengthscale = [[self.hypers["ls"]]]
        WM.covar_module.outputscale = self.hypers["alpha"] ** 2
        WM.likelihood.noise_covar.noise = [self.hypers["sigma"]]

        WM.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
        WM.covar_module.raw_outputscale.requires_grad_(False)
        WM.likelihood.noise_covar.raw_noise.requires_grad_(False)

        return WM