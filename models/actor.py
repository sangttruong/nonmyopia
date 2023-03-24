import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition import (
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
    qKnowledgeGradient,
)
from models.EHIG import qMultiStepHEntropySearch
from models.UncertaintySampling import qUncertaintySampling
from utils.plot import plot_topk

class Actor:
    def __init__(self, parms, buffer):
        self.parms = parms

        # Initialize model
        self.model = SingleTaskGP(buffer.x, buffer.y, covar_module=ScaleKernel(base_kernel=RBFKernel())).to(buffer.x)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        
        if self.parms.algo == "hes":
            self.acqf = qMultiStepHEntropySearch(
                model=self.model,
                parms=parms,
            )
        
        elif self.parms.algo == "us":
            self.acqf = qUncertaintySampling(
                model=self.model,
                parms=parms,
            )
        
        elif self.parms.algo == "kg":
            self.acqf = qKnowledgeGradient(self.model, num_fantasies=self.parms.n_samples)
        
        elif self.parms.algo == "qEI":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            self.acqf = qExpectedImprovement(
                self.model, best_f=buffer.y.max(), sampler=sampler
            )

        elif self.parms.algo == "qPI":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            self.acqf = qProbabilityOfImprovement(
                self.model, best_f=buffer.y.max(), sampler=sampler
            )

        elif self.parms.algo == "qSR":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            self.acqf = qSimpleRegret(self.model, sampler=sampler)

        elif self.parms.algo == "qUCB":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            self.acqf = qUpperConfidenceBound(
                self.model, beta=0.1, sampler=sampler
            )

        elif self.parms.algo == "rs":
            self.acqf = None

        else:
            raise ValueError(f"Unknown algo: {self.parms.algo}")
    

    def query(self):
        """Optimize hes acquisition function, return acq_vals."""
        if self.acqf is None:
            data_x = self.uniform_random_sample_domain(self.parms.domain, 1)
            return data_x[0].reshape(1, -1)
        
        optimizer = torch.optim.Adam(self.acqf.parameters(), lr=self.parms.acq_opt_lr)
        for i in range(self.parms.acq_opt_iter):
            optimizer.zero_grad()
            
            losses = self.acqf.forward()
            loss = losses.sum()
            
            loss.backward()
            optimizer.step()

        # acq_values = self.acqf()
        # best = torch.argmax(acq_values.view(-1), dim=0)
        # next_x = X_design[best]
        next_x = self.acqf.get_next_X()
        
        # self.eval_topk(buffer, next_x, iteration)

        return next_x


    def train(self, buffer):
        model = SingleTaskGP(buffer.x, buffer.y, covar_module=ScaleKernel(base_kernel=RBFKernel())).to(buffer.x)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model.load_state_dict(self.state_dict)
        self.model = model
        self.mll = mll

        # Fit the model
        if not self.parms.learn_hypers:
            print(
                f"config.learn_hypers={self.parms.learn_hypers}, using hypers from config.hypers"
            )
            self.model.covar_module.base_kernel.lengthscale = [
                [self.parms.hypers["ls"]]]
            # NOTE: GPyTorch outputscale should be set to the SynthFunc alpha squared
            self.model.covar_module.outputscale = self.parms.hypers["alpha"] ** 2
            self.model.likelihood.noise_covar.noise = [self.parms.hypers["sigma"]]

            self.model.covar_module.base_kernel.raw_lengthscale.requires_grad_(
                False)
            self.model.covar_module.raw_outputscale.requires_grad_(False)
            self.model.likelihood.noise_covar.raw_noise.requires_grad_(False)

        fit_gpytorch_model(self.mll)

        raw_hypers_str = (
            "\n*Raw GP hypers: "
            f"\nmodel.covar_module.base_kernel.raw_lengthscale={self.model.covar_module.base_kernel.raw_lengthscale.tolist()}"
            f"\nmodel.covar_module.raw_outputscale={self.model.covar_module.raw_outputscale.tolist()}"
            f"\nmodel.likelihood.noise_covar.raw_noise={self.model.likelihood.noise_covar.raw_noise.tolist()}"
        )
        actual_hypers_str = (
            "\n*Actual GP hypers: "
            f"\nmodel.covar_module.base_kernel.lengthscale={self.model.covar_module.base_kernel.lengthscale.tolist()}"
            f"\nmodel.covar_module.outputscale={self.model.covar_module.outputscale.tolist()}"
            f"\nmodel.likelihood.noise_covar.noise={self.model.likelihood.noise_covar.noise.tolist()}"
        )
        print(raw_hypers_str)
        print(actual_hypers_str + "\n")


    def eval(self, data, next_x, iteration):
        """Return evaluation metric."""
        # Initialize X_action
        X_design, X_action = self.acqf.initialize_tensors_topk(data)

        # Set value function
        value_function = self.acqf.value_function_cls(
            model=self.acqf.model,
            sampler=self.acqf.inner_sampler,
            dist_weight=self.parms.dist_weight,
            dist_threshold=self.parms.dist_threshold,
        )

        # Optimize self.acqf_topk
        optimizer = torch.optim.Adam([X_design, X_action], lr=self.parms.acq_opt_lr)
        for i in range(self.parms.acq_opt_iter):
            optimizer.zero_grad()
            losses = -value_function(X_action[:2, :2, :, :]).mean(dim=0)
            loss = losses.sum()
            loss.backward(retain_graph=True)
            optimizer.step()
            X_design.data.clamp_(self.parms.bounds_design[0], self.parms.bounds_design[1])
            X_action.data.clamp_(self.parms.bounds_action[0], self.parms.bounds_action[1])

            if (i+1) % (self.parms.acq_opt_iter//5) == 0 or i == self.parms.acq_opt_iter-1:
                print('Eval:', i+1, loss.item())

        X_action = X_action[:2, :2, :, :]
        acq_values = value_function(X_action)

        optimal_action = X_action[0][0].detach().numpy()
        eval_metric = acq_values[0][0].detach().numpy().tolist()
        print(f'Eval optimal_action: {optimal_action}')

        # Plot optimal_action in special eval plot here
        plot_topk(self.parms, next_x, data, optimal_action, iteration)

        # Return eval_metric and eval_data (or None)
        return eval_metric, optimal_action
