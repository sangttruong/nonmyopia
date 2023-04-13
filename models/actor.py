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
    qMultiStepLookahead
)
from models.EHIG import qMultiStepEHIG
from models.UncertaintySampling import qUncertaintySampling
from models.RandomSampling import qRandomSampling
from utils.plot import draw_losses

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
                nn.Linear(self.parms.hidden_dim*2, self.parms.n_actions * self.parms.x_dim),
                
                Clamp(*self.parms.bounds)
            ]).double().to(self.parms.device)
            
            self._parameters = list(self.map.parameters())
            self.previous_hidden_state = torch.zeros(
                self.parms.n_restarts,
                self.parms.hidden_dim,
                device=self.parms.device
            ).double().to(self.parms.device)
            
        else:
            self.previous_hidden_state = torch.empty(0)
            self.map = []
            
        # Initialize some actor attributes
        self.cost = torch.zeros(1, device=self.parms.device).double()
        self.lookahead_steps = self.parms.lookahead_steps
        self.acqf = self.construct_acqf(WM=WM, maps=self.map, buffer=buffer)
        
    def reset_parameters(self, seed=0, previous_Xy=None):
        print("Resetting actor parameters...")
        torch.manual_seed(seed)
        
        if self.parms.use_amortized_optimization:
            optimizer = torch.optim.AdamW(self._parameters, lr=self.parms.acq_opt_lr)
            
            if previous_Xy is None:
                previous_Xy = torch.rand(self.parms.n_restarts, 
                                        self.parms.x_dim + self.parms.y_dim
                                        ).double().to(self.parms.device) * 2 - 1
            for _ in range(10):
                optimizer.zero_grad()
                return_dict = self.acqf.forward(previous_Xy, 
                                                self.previous_hidden_state, 
                                                self.cost, 
                                                return_X=True)
                
                X_randomized = (torch.rand_like(return_dict["X"][0]) * 0.2 - 0.1) + previous_Xy[:, :self.parms.x_dim]
                loss = torch.mean(torch.pow(return_dict["X"][0] - X_randomized, 2)) # MSE
                
                for i in range(1, self.lookahead_steps):
                    X_randomized = (torch.rand_like(return_dict["X"][i]) * 0.2 - 0.1) + \
                                    return_dict["X"][i-1].detach()[None, ...].expand_as(return_dict["X"][i])
                                    
                    loss += torch.mean(torch.pow(return_dict["X"][i] - X_randomized, 2)) # MSE
                    
                loss.backward()
                optimizer.step()
        else:
            # No need to optimize the parameters of the actor if we are not using amortized optimization.
            self.map = []
            
            for s in range(self.lookahead_steps):
                x = torch.rand(self.parms.n_samples**s * self.parms.n_restarts,
                           self.parms.x_dim,
                           device=self.parms.device).double()
                self.map.append((x * 2 - 1).requires_grad_(True))
                
            a = torch.rand((self.parms.n_samples**self.lookahead_steps * self.parms.n_restarts) * self.parms.n_actions,
                            self.parms.x_dim, 
                            device=self.parms.device).double()
            self.map.append((a * 2 - 1).requires_grad_(True)) 
            self._parameters = self.map
            
            if self.parms.algo == "HES":
                self.acqf.update_maps(self.map)

    def construct_acqf(self, WM, maps, buffer=None):
        if self.parms.algo == "HES":
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
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            return qRandomSampling(model=WM, sampler=sampler)
        
        elif self.parms.algo == "qMSL":
            
            return qMultiStepLookahead(
                model=WM,
                batch_sizes=[1 for _ in range(self.lookahead_steps)],
                num_fantasies=[self.parms.n_samples for _ in range(self.lookahead_steps)]
            )

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
            
        # Reset the actor parameters for diversity
        self.reset_parameters(seed=iteration, previous_Xy=previous_Xy)
        
        # Optimize the acquisition function
        optimizer = torch.optim.AdamW(self._parameters, lr=self.parms.acq_opt_lr)
        lr_scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
                                                   
        best_results = {}
        best_loss = float("inf")
        losses = []
        early_stop_counter = 0
        
        for opt_iter in range(self.parms.acq_opt_iter):
            optimizer.zero_grad()
            
            if self.parms.algo == "HES":
                return_dict = self.acqf.forward(
                    previous_Xy=previous_Xy,
                    previous_hidden_state=self.previous_hidden_state,
                    previous_cost=self.cost,
                    return_first=True,
                    return_actions=True
                )
                loss = - return_dict["acqf_values"].sum()
            else:
                X = torch.cat(self.map, dim=0)
                acqf_values = self.acqf.forward(X=X)
                loss = - acqf_values.sum()
                
            losses.append(loss.item())
            print('Loss:', loss.item())
            
            # Get the best results
            if opt_iter >= self.parms.acq_warmup_iter:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    # We need to follow the plan step by step
                    if self.parms.algo == "HES":
                        best_results = self.acqf.get_next_X_and_optimal_actions(return_dict, previous_Xy, self.cost)
                    else:
                        best_results = {}
                        X = torch.cat(self.map, dim=0)
                        best_results["next_X"] = self.acqf.get_multi_step_tree_input_representation(X)[0].cpu().detach()
                        best_results["acqf_values"] = acqf_values.cpu().detach()
                        best_results["optimal_actions"] = self.map[-1].cpu().detach()
                        best_results["hidden_state"] = self.previous_hidden_state
                        
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter > self.parms.acq_earlystop_iter:
                        print("Early stopped at epoch", opt_iter)
                        break
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
                
        # Update the hidden state
        self.previous_hidden_state = best_results["hidden_state"].to(self.parms.device)
        
        # Update new cost
        # self.cost = best_results["cost"]
        
        # Draw losses by acq_opt_iter using matplotlib
        draw_losses(config=self.parms, 
                    losses=losses, 
                    iteration=iteration)
        
        return best_results["next_X"], best_results["optimal_actions"], best_results["acqf_values"]

    def eval(self, buffer):
        previous_x = buffer.x[-self.parms.n_restarts:]
        previous_y = buffer.y[-self.parms.n_restarts:]
        previous_Xy = torch.cat((previous_x, previous_y), dim=-1).to(self.parms.device)
        
        # Reset the actor parameters for diversity
        self.reset_parameters(seed=0, previous_Xy=previous_Xy)
        
        # Optimize the acquisition function
        optimizer = torch.optim.AdamW(self._parameters, lr=self.parms.acq_opt_lr)
        lr_scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
                                                   
        best_results = {}
        best_loss = float("inf")
        losses = []
        early_stop_counter = 0
        
        for opt_iter in range(self.parms.acq_opt_iter):
            optimizer.zero_grad()
            
            if self.parms.algo == "HES":
                return_dict = self.acqf.forward(
                    previous_Xy=previous_Xy,
                    previous_hidden_state=self.previous_hidden_state,
                    previous_cost=self.cost,
                    return_first=True,
                    return_actions=True
                )
                loss = - return_dict["acqf_values"].sum()
            else:
                X = torch.cat(self.map, dim=0)
                acqf_values = self.acqf.forward(X=X)
                loss = - acqf_values.sum()
                
            losses.append(loss.item())
            print('Loss:', loss.item())
    
    def set_WM(self, WM):
        self.acqf.model = WM
        
    def set_lookahead_steps(self, lookahead_steps):
        self.lookahead_steps = lookahead_steps
        if self.parms.algo == "HES":
            self.acqf.lookahead_steps = lookahead_steps
        elif self.parms.algo == "qMSL":
            model = self.acqf.model
            self.acqf = qMultiStepLookahead(
                model=model,
                batch_sizes=[1 for _ in range(self.lookahead_steps)],
                num_fantasies=[self.parms.n_samples for _ in range(self.lookahead_steps)]
            )
        
        if not self.parms.use_amortized_optimization:
            self.map = []
        
class Clamp(nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        if torch.any(x > self.max_value) or torch.any(x < self.min_value):
            return torch.clamp(x, self.min_value, self.max_value)
        return x