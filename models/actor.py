import torch
import torch.nn as nn
import numpy as np
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition import (
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
    qKnowledgeGradient,
)
from models.EHIG import qMultiStepEHIG
from models.UncertaintySampling import qUncertaintySampling
from utils.plot import draw_posterior

class Actor:
    def __init__(self, parms, WM, buffer):
        self.parms = parms

        if self.parms.use_amortized_optimization:
            self.map = nn.ModuleList([
                nn.Linear(self.parms.x_dim + self.parms.y_dim, self.parms.hidden_dim), 
                nn.Linear(self.parms.hidden_dim, self.parms.hidden_dim),
                nn.GRUCell(self.parms.hidden_dim, self.parms.hidden_dim),
                
                nn.Linear(self.parms.hidden_dim*2, self.parms.hidden_dim),
                nn.Linear(self.parms.hidden_dim*2, self.parms.x_dim),
                
                nn.Linear(self.parms.hidden_dim*2, self.parms.hidden_dim),
                nn.Linear(self.parms.hidden_dim*2, self.parms.n_actions * self.parms.x_dim)
            ]).double().to(self.parms.device)
            
            self._parameters = list(self.map.parameters())
            self.previous_hidden_state = torch.rand(
                self.parms.n_restarts,
                self.parms.hidden_dim,
                device=self.parms.device
            ).double()
            # self.reset_parameters(1)
        else:
            self.previous_hidden_state = torch.empty(0)
            self.map = [
                torch.rand(self.parms.n_samples**s, 
                           self.parms.x_dim, 
                           device=self.parms.device).double().requires_grad_(True) \
                for s in range(self.parms.lookahead_steps)
            ]
            self.map.append(torch.rand((self.parms.n_samples**self.parms.lookahead_steps) * self.parms.n_actions,
                                       self.parms.x_dim, 
                                       device=self.parms.device).double().requires_grad_(True))
            self._parameters = self.map
            
        self.lookahead_steps = self.parms.lookahead_steps
        self.acqf = self.construct_acqf(WM=WM, maps=self.map, buffer=buffer)

    def reset_parameters(self, seed=0):
        torch.manual_seed(seed)
        for p in self._parameters:
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                nn.init.normal_(p)

    def construct_acqf(self, WM, maps, buffer=None):
        if self.parms.algo == "hes":
            return qMultiStepEHIG(
                model=WM,
                parms=self.parms,
                maps=maps,
            )
        
        elif self.parms.algo == "us":
            return qUncertaintySampling(
                model=WM,
                parms=self.parms,
            )
        
        elif self.parms.algo == "kg":
            return qKnowledgeGradient(
                model=WM, 
                num_fantasies=self.parms.n_samples
            )
        
        elif self.parms.algo == "qEI":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            return qExpectedImprovement(
                model=WM, 
                best_f=buffer.y.max(), 
                sampler=sampler
            )

        elif self.parms.algo == "qPI":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            return qProbabilityOfImprovement(
                model=WM, best_f=buffer.y.max(), sampler=sampler
            )

        elif self.parms.algo == "qSR":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            return qSimpleRegret(model=WM, sampler=sampler)

        elif self.parms.algo == "qUCB":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            return qUpperConfidenceBound(
                model=WM, beta=0.1, sampler=sampler
            )

        elif self.parms.algo == "rs":
            return None

        else:
            raise ValueError(f"Unknown algo: {self.parms.algo}")
    

    def query(self, buffer, iteration):
        if self.acqf is None:
            data_x = self.uniform_random_sample_domain(self.parms.domain, 1)
            return data_x[0].reshape(1, -1)
        
        previous_x = buffer.x[-self.parms.n_restarts:]
        previous_y = buffer.y[-self.parms.n_restarts:]
        previous_Xy = torch.cat((previous_x, previous_y), dim=-1).to(self.parms.device)
        
        if abs(torch.rand(1).item()) > self.parms.epsilon:
            self.previous_hidden_state = torch.rand(
                self.parms.n_restarts,
                self.parms.hidden_dim,
                device=self.parms.device
            ).double()
            previous_Xy = torch.rand_like(previous_Xy).double().to(self.parms.device)
            
            
        # Optimize the acquisition function
        # self.previous_hidden_state = self.previous_hidden_state.requires_grad_(True)
        # previous_Xy = previous_Xy.requires_grad_(True)
        optimizer = torch.optim.Adam(self._parameters, lr=self.parms.acq_opt_lr)
        
        for _ in range(self.parms.acq_opt_iter):
            optimizer.zero_grad()
            return_dict = self.acqf.forward(previous_Xy, self.previous_hidden_state)
            
            loss = - return_dict["acqf_values"].sum()
            loss.backward()
            optimizer.step()
        
            print('Loss:', loss.item())

        # We need to follow the plan step by step
        results = self.acqf.get_next_X_and_optimal_actions(previous_Xy, self.previous_hidden_state)
        
        # Update the hidden state
        self.previous_hidden_state = results["hidden_state"]
        
        # breakpoint()
        # Draw posterior in imagination
        # draw_posterior(config=self.parms,
        #                model=self.acqf.model,
        #                train_x=buffer.x,
        #                iteration=iteration,
        #                mode="imagination")
        
        return results["next_X"], results["optimal_actions"], results["acqf_values"]

    def set_WM(self, WM):
        self.acqf.model = WM
        
    def set_lookahead_steps(self, lookahead_steps):
        self.lookahead_steps = lookahead_steps
        self.acqf.lookahead_steps = lookahead_steps
        
        if not self.parms.use_amortized_optimization:
            self.map = [
                torch.rand(self.parms.n_samples**s, 
                           self.parms.x_dim, 
                           device=self.parms.device).double().requires_grad_(True) \
                for s in range(self.lookahead_steps)
            ]
            self.map.append(torch.rand((self.parms.n_samples**self.lookahead_steps) * self.parms.n_actions,
                                       self.parms.x_dim, 
                                       device=self.parms.device).double().requires_grad_(True))
            self._parameters = self.map
            
            self.acqf.update_maps(self.map)
        
class Clamp(nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, self.min_value, self.max_value)