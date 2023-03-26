import torch
import torch.nn as nn
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

class Actor:
    def __init__(self, parms, WM, buffer):
        self.parms = parms

        if self.parms.use_amortized_optimization:
            self.rnn_map = nn.ModuleList([
                nn.Linear(self.parms.x_dim + self.parms.y_dim, self.parms.hidden_dim), 
                nn.ReLU(),
                nn.GRUCell(self.parms.hidden_dim, self.parms.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.parms.hidden_dim, self.parms.x_dim),
                # Project to domain layer
                
            ]).double().to(self.parms.device)
            
            self.lin_map = nn.Linear(self.parms.hidden_dim, self.parms.n_actions * self.parms.x_dim).double().to(self.parms.device)
            self._parameters = list(self.rnn_map.parameters()) + list(self.lin_map.parameters())
            
            self.previous_hidden_state = torch.zeros(
                self.parms.n_restarts,
                self.parms.hidden_dim,
                device=self.parms.device
            ).double()
        else:
            # self.map = torch.rand(...)
            # self._parameters = torch.Parameters(self.map)
            pass

        self.optimizer = torch.optim.Adam(self._parameters, lr=self.parms.acq_opt_lr)
        
        self.acqf = self.construct_acqf(WM=WM, maps=[self.rnn_map, self.lin_map], buffer=buffer)


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
    

    def query(self, previous_Xy):
        if self.acqf is None:
            data_x = self.uniform_random_sample_domain(self.parms.domain, 1)
            return data_x[0].reshape(1, -1)
        
        previous_Xy = (previous_Xy[0].to(self.parms.device), previous_Xy[1].to(self.parms.device))
        previous_Xy = torch.cat(previous_Xy, dim=-1)
        
        if abs(torch.rand(1).item()) > self.parms.epsilon:
            self.previous_hidden_state = torch.rand(
                self.parms.n_restarts,
                self.parms.hidden_dim,
                device=self.parms.device
            ).double()
            previous_Xy = torch.rand_like(previous_Xy).double().to(self.parms.device)
        
        for _ in range(self.parms.acq_opt_iter):
            self.optimizer.zero_grad()
            loss = - self.acqf.forward(previous_Xy.clone(), self.previous_hidden_state.clone()).sum()
            print('Loss:', loss.item())
            loss.backward()
            self.optimizer.step()
        
        # next_x, hidden_state = self.acqf.get_next_X(previous_Xy, self.previous_hidden_state)
        next_x, hidden_state = self.acqf.get_next_X_in_topK(previous_Xy, self.previous_hidden_state)
    
        # Update hidden state
        self.previous_hidden_state = hidden_state
            
        return next_x
    
    def get_topK_actions(self, previous_Xy):
        previous_Xy = (previous_Xy[0].to(self.parms.device), previous_Xy[1].to(self.parms.device))
        previous_Xy = torch.cat(previous_Xy, dim=-1)
        return self.acqf.get_topK_actions(previous_Xy, self.previous_hidden_state)

    def set_WM(self, WM):
        self.acqf.model = WM