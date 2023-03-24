import torch
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
from typing import Any, Optional
from botorch.acquisition import MCAcquisitionObjective
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    PosteriorTransform,
)


class qMultiStepHEntropySearch(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        parms: Any = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=parms.n_samples, collapse_batch_dims=True)
        
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        
        self.parms = parms
        
        if self.parms.use_amortized_optimization:
            self.map = nn.ModuleList([
                nn.Linear(self.parms.x_dim + self.parms.y_dim, self.parms.hidden_dim), 
                nn.ReLU(),
                nn.GRUCell(self.parms.hidden_dim, self.parms.hidden_dim),#, batch_first=True),
                # SelectItem(0), # Select all output from GRU
                nn.ReLU(),
                nn.Linear(self.parms.hidden_dim, self.parms.x_dim),
            ]).double().to(self.parms.device)
            
            self.linear = nn.Linear(self.parms.x_dim, self.parms.x_dim).double().to(self.parms.device)
            self._parameters = self.map.parameters()
            
        else:
            # self.map = torch.rand(...)
            # self._parameters = torch.Parameters(self.map)
            pass
    
    def parameters(self):
        return self._parameters
        
    def get_optimization_variables(self, step: int, X_y = None, hidden_state=None):
        if self.parms.use_amortized_optimization:
            out = self.map[0](X_y)
            out = self.map[1](out)
            out_hidden = self.map[2](out, hidden_state)
            out = self.map[3](out_hidden)
            out = self.map[4](out)
            return out, out_hidden
        else:
            # return X[step]
            pass                       
        
    def forward(self):
        #X: tensorDict
        # number of keys = number of lookahead steps
        
        fantasized_model = self.model
        previous_XY = torch.zeros(
            self.parms.n_restarts, 
            1, 
            self.parms.x_dim + self.parms.y_dim, 
            device=self.parms.device
        ).double() # x[-1]
        # >>> (n_samples * n_restarts) * seq_length * dim
        
        previous_hidden_state = torch.zeros(
            self.parms.n_restarts,
            self.parms.hidden_dim,
            device=self.parms.device
        ).double()
        
        for step in tqdm(range(self.parms.lookahead_steps), desc='Looking ahead'):
            # condition on X[step], then sample, then condition on (x, y)
            X, hidden_state = self.get_optimization_variables(
                step=step, X_y=previous_XY[:, -1, :], hidden_state=previous_hidden_state
            )
            new_shape = [self.parms.n_samples] * step + [self.parms.n_restarts] + [self.parms.x_dim]
            X = X.reshape(*new_shape)
            # >>> num_x_{step} * x_dim
            
            ppd = fantasized_model.posterior(X)
            ys = self.sampler(ppd)
            # >>> n_samples * num_x_{step} * y_dim
            
            X_expanded_shape = [ys.shape[0]] + [-1] * (len(new_shape) + 1)
            X_expanded = X[None, ..., None, :].expand(*X_expanded_shape)
            ys_expanded = ys[..., None, :]
            X_y = torch.cat((X_expanded, ys_expanded), dim=-1)
            # >>> n_samples * num_x_{step} * 1 * y_dim
            
            previous_XY_shape = [1] + [self.parms.n_samples]*step + [self.parms.n_restarts, previous_XY.shape[-2], previous_XY.shape[-1]]
            previous_XY = previous_XY[None, ...].reshape(*previous_XY_shape)
            previous_XY_shape[0] = self.parms.n_samples
            previous_XY = previous_XY.expand(*previous_XY_shape)
            # >>> n_samples * num_x_{step} * prev_seq_length * y_dim
            previous_XY = torch.cat((previous_XY, X_y), dim=-2)
            # >>> n_samples * num_x_{step} * (prev_seq_length + 1) * y_dim
            previous_XY = previous_XY.view(-1, step + 2, self.parms.x_dim + self.parms.y_dim)
            # >>> (n_samples * num_x_{step}) * seq_length * y_dim
            
            # Update conditions
            fantasized_model = fantasized_model.condition_on_observations(
                X,
                ys
            )
            
            # Update hidden state
            previous_hidden_state = hidden_state[None, ...].expand(self.parms.n_samples, -1, -1)
            previous_hidden_state = previous_hidden_state.reshape(-1, self.parms.hidden_dim)
            
            # Question: How to condition on multiple observations?
            # E.g. {x_0, y_00}, {x_0, y_01}, {x_0, y_02}..
            # ==> D_00 = D U {x_0, y_00},
            #     D_01 = D U {x_0, y_01}, 
            #     D_02 = D U {x_0, y_02}
            
        # Compute Top-K
        actions, _ = self.get_optimization_variables(
            step=self.parms.lookahead_steps, X_y=previous_XY[:, -1, :], hidden_state=previous_hidden_state
        )
        actions = self.linear(actions).view(-1, self.parms.x_dim)
        
        return self.parms.compute_expectedloss_function(
            model=fantasized_model, 
            actions=actions, 
            sampler=self.sampler, 
            info=self.parms
        )
        # >>> batch number of x_0
        
    def get_next_X(self):
        # x[-1]
        previous_XY = torch.zeros(
            self.parms.n_restarts, 
            1, 
            self.parms.x_dim + self.parms.y_dim, 
            device=self.parms.device
        ).double()
        previous_hidden_state = torch.zeros(
            self.parms.n_restarts,
            self.parms.hidden_dim,
            device=self.parms.device
        ).double()
        X = self.get_optimization_variables(step=0, X_y=previous_XY[:, -1, :], hidden_state=previous_hidden_state)
        
        return X
    
    
class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

