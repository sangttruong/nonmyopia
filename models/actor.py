from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.acquisition import (
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.sampling.normal import SobolQMCNormalSampler
import torch
import os
import numpy as np
from tqdm import tqdm
from models.ehig_acqf import custom_warmstart_multistep
from models.ehig_acqf_new import qMultiStepHEntropySearch
from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead
from botorch.generation.gen import get_best_candidates

class Actor:
    def __init__(self, parms, data):
        self.parms = parms
        self.seed = parms.seed
        self.algo = parms.algo

        # Initialize model
        self.mll, self.model = self.initialize_model(
            data, covar_module=ScaleKernel(base_kernel=RBFKernel())
        )
            
        self.acqf = None
        self.full_optimizer = None
        
        if self.parms.algo in ["hes_mc", "hes_vi"]:
            self.acqf = qMultiStepHEntropySearch(
                model=self.model,
                parms=parms
            )
            
        elif self.parms.algo == "rs":
            pass
        
        elif self.parms.algo == "us":
            pass
        
        elif self.parms.algo == "kg" or self.parms.algo == "kgtopk":
            pass 
        
        elif self.parms.algo in ["qEI", "qPI", "qSR", "qUCB"]:
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            if self.parms.algo == "qEI":
                self.acqf = qExpectedImprovement(
                    self.model, best_f=data.y.max(), sampler=sampler
                )
            elif self.parms.algo == "qPI":
                self.acqf = qProbabilityOfImprovement(
                    self.model, best_f=data.y.max(), sampler=sampler
                )
            elif self.parms.algo == "qSR":
                self.acqf = qSimpleRegret(self.model, sampler=sampler)
            elif self.parms.algo == "qUCB":
                self.acqf = qUpperConfidenceBound(
                    self.model, beta=0.1, sampler=sampler
                )


    def print_model_hypers(self, model):
        """Print current hyperparameters of GP model."""
        raw_hypers_str = (
            "\n*Raw GP hypers: "
            f"\nmodel.covar_module.base_kernel.raw_lengthscale={model.covar_module.base_kernel.raw_lengthscale.tolist()}"
            f"\nmodel.covar_module.raw_outputscale={model.covar_module.raw_outputscale.tolist()}"
            f"\nmodel.likelihood.noise_covar.raw_noise={model.likelihood.noise_covar.raw_noise.tolist()}"
        )
        actual_hypers_str = (
            "\n*Actual GP hypers: "
            f"\nmodel.covar_module.base_kernel.lengthscale={model.covar_module.base_kernel.lengthscale.tolist()}"
            f"\nmodel.covar_module.outputscale={model.covar_module.outputscale.tolist()}"
            f"\nmodel.likelihood.noise_covar.noise={model.likelihood.noise_covar.noise.tolist()}"
        )
        print(raw_hypers_str)
        print(actual_hypers_str + "\n")


    def initialize_model(self, data, state_dict=None, covar_module=None):
        model = SingleTaskGP(data.x, data.y, covar_module=covar_module).to(data.x)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model


    def uniform_random_sample_domain(self, domain, n, config):
        """Draws a sample uniformly at random from domain (a list of tuple bounds)."""
        list_of_arr_per_dim = [np.random.uniform(
            dom[0], dom[1], n) for dom in domain]
        return torch.tensor(
            np.array(
                list_of_arr_per_dim).T, device=config.device, dtype=config.torch_dtype
        )
    

    def query(self):
        if self.parms.algo == "hes_mc" or self.parms.algo == "hes_vi":
            next_x = self.optimize_hes()
              
        elif self.parms.algo == "rs":
            next_x = self.optimize_rs()
            
        elif self.parms.algo == "us":
            next_x = self.optimize_us()
            
        elif self.parms.algo == "kg" or self.parms.algo == "kgtopk":
            pass 

        elif self.parms.algo == "random":
            next_x = self.parms.bounds[0] + (
                self.parms.bounds[1] - self.parms.bounds[0]
            ) * torch.rand([self.parms.n_candidates, self.parms.n_dim], device=self.parms.device)
            

        elif self.parms.algo in ["qEI", "qPI", "qSR", "qUCB"]:
            # to keep the restart conditions the same
            torch.manual_seed(seed=0)
            bounds = torch.tensor(
                [
                    [self.parms.bounds[0]] * self.parms.n_dim,
                    [self.parms.bounds[1]] * self.parms.n_dim,
                ]
            ).to(self.parms.device).double()
            next_x = self.optimize_acqf_and_get_suggested_point(bounds)

        return next_x
    
    
    def optimize_acqf_and_get_suggested_point(self, bounds):
        """Optimizes the acquisition function, and returns the candidate solution."""
        is_ms = isinstance(self.acqf, qMultiStepLookahead)
        input_dim = bounds.shape[1]
        q = self.acqf.get_augmented_q_batch_size(self.parms.batch_size) if is_ms else self.parms.batch_size
        raw_samples = 200 * input_dim * self.parms.batch_size # 200 * 2 * 1
        
        #if is_ms:
            #raw_samples *= (len(algo_params.get("lookahead_n_fantasies")) + 1)
            #num_restarts *=  (len(algo_params.get("lookahead_n_fantasies")) + 1)
            
        if self.full_optimizer is not None:
            batch_initial_conditions = custom_warmstart_multistep(
                acq_function=self.acqf,
                bounds=bounds,
                num_restarts=self.parms.n_restarts,
                raw_samples=raw_samples,
                n_lookahead_steps=self.parms.lookahead_steps,
                full_optimizer=self.full_optimizer,
            )
        else:
            batch_initial_conditions = None
        
        candidates, acq_values = optimize_acqf(
                acq_function=self.acqf,
                bounds=bounds,
                q=q,
                num_restarts=self.parms.n_restarts,
                raw_samples=raw_samples,
                options={},
                batch_initial_conditions=batch_initial_conditions,
                return_best_only=False,
                return_full_tree=is_ms
            )

        candidates = candidates.detach()
        if is_ms:
            # save all tree variables for multi-step initialization
            self.full_optimizer = candidates.clone()
            candidates = self.acqf.extract_candidates(candidates)

        new_x = get_best_candidates(batch_candidates=candidates, batch_values=acq_values)
        
        return new_x


    def train(self, data):
        self.mll, self.model = self.initialize_model(
            data,
            self.model.state_dict(),
            covar_module=ScaleKernel(base_kernel=RBFKernel()),
        )
        
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
        self.print_model_hypers(self.model)
        
    def optimize_hes(self):
        """Optimize hes acquisition function, return acq_vals."""
        optimizer = torch.optim.Adam(self.acqf.parameters(), lr=self.parms.acq_opt_lr)
        for i in range(self.parms.acq_opt_iter):
            optimizer.zero_grad()
            
            losses = self.acqf.forward() # l = -f(a)
            loss = losses.sum()
            
            loss.backward(retain_graph=True)
            optimizer.step()

        # acq_values = self.acqf()
        # best = torch.argmax(acq_values.view(-1), dim=0)
        # next_x = X_design[best]
        next_x = self.acqf.get_next_X()
        return next_x
    
    def optimize_rs(self):
        """Optimize random search (rs) acquisition function, return next_x."""
        data_x = self.uniform_random_sample_domain(self.parms.domain, 1)
        next_x = data_x[0].reshape(1, -1)
        return next_x


    def optimize_us(self):
        """Optimize uncertainty sampling (us) acquisition function, return next_x."""
        n_acq_opt_samp = 500
        data_x = self.uniform_random_sample_domain(self.parms.domain, n_acq_opt_samp)
        acq_values = self.model(data_x).variance
        best = torch.argmax(acq_values.view(-1), dim=0)
        next_x = data_x[best].reshape(1, -1)
        return next_x


    def optimize_kg(self, batch_x0s, batch_a1s, model, sampler, config, iteration):
        """Optimize knowledge gradient (kg) acquisition function, return next_x."""
        # if not config.n_dim == config.n_dim_action:
        #     batch_a1s = initialize_action_tensor_kg(config)


        # optimizer = torch.optim.Adam([batch_x0s, batch_a1s], lr=config.acq_opt_lr)
        # for i in range(config.acq_opt_iter):
        #     losses = -qkg(batch_x0s, batch_a1s)
        #     loss = losses.sum()
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     batch_x0s.data.clamp_(config.bounds[0], config.bounds[1])
        #     batch_a1s.data.clamp_(config.bounds_action[0], config.bounds_action[1])
        #     if (i + 1) % (config.acq_opt_iter // 5) == 0 or i == config.acq_opt_iter - 1:
        #         print(iteration, i + 1, loss.item())
        # acq_values = qkg(batch_x0s, batch_a1s)
        # best = torch.argmax(acq_values.view(-1), dim=0)
        # next_x = batch_x0s[best]
        # return next_x
        pass
