import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from botorch.sampling.normal import SobolQMCNormalSampler
from utils.utils import sang_sampler

class qMultiStepEHIG(nn.Module):
    def __init__(self, model, parms, maps) -> None:
        super().__init__()
        self.parms = parms
        self.model = model
        self.lookahead_steps = parms.lookahead_steps
        # self.sampler = SobolQMCNormalSampler(
        #     sample_shape=parms.n_samples, collapse_batch_dims=True
        # )
        self.sampler = sang_sampler(num_samples=parms.n_samples)
        
        self.maps = maps
        
    def update_maps(self, maps):
        self.maps = maps
                    
    def get_optimization_variables(self, step: int, Xy = None, hidden_state=None):
        if self.parms.use_amortized_optimization:
            out1 = self.maps[0](Xy)
            out1 = torch.relu(out1)
            
            # out1 = torch.tanh(out1)
            
            out2 = self.maps[1](out1)
            # out2 = torch.tanh(out2)
            
            out_hidden = self.maps[2](out2, hidden_state)
            
            out3 = self.maps[3](torch.cat((out_hidden, out2), dim=-1))
            out3 = torch.relu(out3)

            
            out4 = self.maps[4](torch.cat((out3, out1), dim=-1))
            return out4, out_hidden
        
        else:
            return self.maps[step], hidden_state
        
    def get_actions(self, Xy = None, hidden_state=None):       
        if self.parms.use_amortized_optimization:
            out1 = self.maps[0](Xy)
            out1 = torch.relu(out1)
            
            out2 = self.maps[1](out1)
            # out2 = torch.tanh(out2)
            
            out_hidden = self.maps[2](out2, hidden_state)
            
            out3 = self.maps[5](torch.cat((out_hidden, out2), dim=-1))
            out3 = torch.relu(out3)
            
            out4 = self.maps[6](torch.cat((out3, out1), dim=-1))
            return out4, out_hidden
        else:
            return self.maps[self.lookahead_steps], hidden_state          
    
    def constraint(self, X, prev_X, neighbor_size=0.1):
        X = torch.sigmoid(X) * (neighbor_size*2) + (prev_X - neighbor_size)
        if torch.any(X < -1) or torch.any(X > 1):
            return torch.clamp(X, -1, 1)
        return X
    
    def forward(self, previous_Xy, previous_hidden_state, return_first=False, return_actions=False):
        # previous_Xy = torch.rand(
        #     self.parms.n_restarts, 
        #     self.parms.x_dim + self.parms.y_dim, 
        #     device=self.parms.device
        # ).double()
        # # >>> (n_samples * n_restarts) * seq_length * dim
        
        # previous_hidden_state = torch.zeros(
        #     self.parms.n_restarts,
        #     self.parms.hidden_dim,
        #     device=self.parms.device
        # ).double()
        
        previous_Xy = previous_Xy.reshape(self.parms.n_restarts, self.parms.x_dim + self.parms.y_dim)
        # if not (return_first and return_actions):
        #     fantasized_model = copy.deepcopy(self.model)
        # else:
        #     fantasized_model = self.model
            
        fantasized_model = copy.deepcopy(self.model)
        first_X = first_hidden_state = None
        
        for step in tqdm(range(self.lookahead_steps), desc='Looking ahead'):
            # condition on X[step], then sample, then condition on (x, y)
            # Pass through RNN
            X, hidden_state = self.get_optimization_variables(
                step=step, Xy=previous_Xy, hidden_state=previous_hidden_state
            )
            
            X = self.constraint(X, previous_Xy[:, :self.parms.x_dim])
            
            if step == 0 and return_first:
                first_X = X
                first_hidden_state = hidden_state
                
            new_shape = [self.parms.n_samples] * step + [self.parms.n_restarts, self.parms.x_dim]
            X = X.reshape(*new_shape)
            # >>> num_x_{step} * x_dim
            
            # Sample posterior
            ppd = fantasized_model.posterior(X)
            ys = self.sampler(ppd)
            # >>> n_samples * num_x_{step} * y_dim
            
            X_expanded_shape = [ys.shape[0]] + [-1] * (len(new_shape))
            X_expanded = X[None, ...].expand(*X_expanded_shape)
            Xy = torch.cat((X_expanded, ys), dim=-1)
            # >>> n_samples * num_x_{step} * 1 * dim
            
            # Update previous_Xy
            previous_Xy = Xy.reshape(-1, self.parms.x_dim + self.parms.y_dim)
            # >>> (n_samples * num_x_{step}) * seq_length * y_dim
            
            # Update conditions
            fantasized_model = fantasized_model.condition_on_observations(
                X,
                ys
            )
            
            # Update hidden state
            previous_hidden_state = hidden_state[None, ...].expand(self.parms.n_samples, -1, -1)
            previous_hidden_state = previous_hidden_state.reshape(-1, self.parms.hidden_dim)
        
        # Compute Top-K actions
        actions, hidden_state = self.get_actions(Xy=previous_Xy, hidden_state=previous_hidden_state)
        actions = actions.reshape(-1, self.parms.n_actions, self.parms.x_dim)
        actions = [self.constraint(actions[:, i], previous_Xy[:, :self.parms.x_dim])[..., None, :]
                   for i in range(self.parms.n_actions)]
        actions = torch.cat(actions, dim=-2)
        
        new_shape = [self.parms.n_samples] * self.lookahead_steps \
                  + [self.parms.n_restarts, self.parms.n_actions, self.parms.x_dim]
        actions = actions.reshape(*new_shape)
        
        acqf_values = self.parms.compute_expectedloss_function(
            model=fantasized_model, 
            actions=actions, 
            sampler=self.sampler, 
            info=self.parms
        )
        # >>> batch number of x_0
        
        return_dict = {}
        return_dict['acqf_values'] = acqf_values
        
        if return_first:
            return_dict['first_X'] = first_X
            return_dict['first_hidden_state'] = first_hidden_state
        
        if return_actions:
            return_dict['actions'] = actions
        
        return return_dict
        
    def get_next_X_and_optimal_actions(self, previous_Xy, previous_hidden_state):
        return_dict = self.forward(
            previous_Xy=previous_Xy,
            previous_hidden_state=previous_hidden_state,
            return_first=True,
            return_actions=True
        )
        
        selected_restart = torch.argmax(return_dict["acqf_values"])
        next_X = return_dict["first_X"][selected_restart].reshape(1, self.parms.x_dim)
        hidden_state = return_dict["first_hidden_state"][selected_restart:selected_restart+1].expand(self.parms.n_restarts, -1)
        
        topK_actions = return_dict["actions"][..., selected_restart, :, :].reshape(-1, self.parms.n_actions, self.parms.x_dim)    
        topK_values = return_dict["acqf_values"][selected_restart].reshape(-1, 1)
        
        return {
            "next_X": next_X.detach(),
            "hidden_state": hidden_state.detach(),
            "optimal_actions": topK_actions.detach(),
            "acqf_values": topK_values.detach(),
            "selected_restart": selected_restart.detach()
        }
        
