import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from botorch.sampling.normal import SobolQMCNormalSampler


class qMultiStepEHIG(nn.Module):
    def __init__(self, model, parms, maps) -> None:
        super().__init__()
        self.parms = parms
        self.model = model
        self.sampler = SobolQMCNormalSampler(
            sample_shape=parms.n_samples, collapse_batch_dims=True
        )
        self.rnn_map, self.lin_map = maps
                    
    def get_optimization_variables(self, step: int, Xy = None, hidden_state=None):
        if self.parms.use_amortized_optimization:
            out = self.rnn_map[0](Xy)
            out = self.rnn_map[1](out)
            out_hidden = self.rnn_map[2](out, hidden_state)
            out = self.rnn_map[3](out_hidden)
            out = self.rnn_map[4](out)
            # out = torch.tanh(out)
            return out, out_hidden
        else:
            # return X[step]
            pass                       

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
        fantasized_model = copy.deepcopy(self.model)
        hidden_state = None
        first_X = first_hidden_state = None
        
        for step in tqdm(range(self.parms.lookahead_steps), desc='Looking ahead'):
            # condition on X[step], then sample, then condition on (x, y)
            X, hidden_state = self.get_optimization_variables(
                step=step, Xy=previous_Xy, hidden_state=previous_hidden_state
            )
            if step == 0 and return_first:
                first_X = X.clone()
                first_hidden_state = hidden_state.clone()
                
            new_shape = [self.parms.n_samples] * step + [self.parms.n_restarts, self.parms.x_dim]
            X = X.reshape(*new_shape).detach()
            # >>> num_x_{step} * x_dim
            
            ppd = fantasized_model.posterior(X)
            ys = self.sampler(ppd).detach()
            # >>> n_samples * num_x_{step} * y_dim
            
            X_expanded_shape = [ys.shape[0]] + [-1] * (len(new_shape))
            X_expanded = X[None, ...].expand(*X_expanded_shape)
            Xy = torch.cat((X_expanded, ys), dim=-1)
            # >>> n_samples * num_x_{step} * 1 * dim
            
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
        
        # Compute Top-K
        _, hidden_state = self.get_optimization_variables(
            step=self.parms.lookahead_steps, Xy=previous_Xy, hidden_state=previous_hidden_state
        )
        
        actions = self.lin_map(hidden_state)
        # actions = torch.tanh(actions)
        new_shape = [self.parms.n_samples] * self.parms.lookahead_steps \
                  + [self.parms.n_restarts, self.parms.n_actions, self.parms.x_dim]
        actions = actions.reshape(*new_shape)
        
        acqf_values = self.parms.compute_expectedloss_function(
            model=fantasized_model, 
            actions=actions, 
            sampler=self.sampler, 
            info=self.parms
        )
        # >>> batch number of x_0
        
        if return_first:
            return acqf_values, first_X, first_hidden_state
        
        elif return_actions:
            new_hidden_shape = [self.parms.n_samples] * self.parms.lookahead_steps \
                                + [self.parms.n_restarts, self.parms.hidden_dim]
            return acqf_values, actions, hidden_state.reshape(*new_hidden_shape)
        
        return acqf_values.mean(-1) # Average over top-K
        
    def get_next_X(self, previous_Xy, previous_hidden_state):
        acqf_values, X, hidden_state = self.forward(
            previous_Xy=previous_Xy,
            previous_hidden_state=previous_hidden_state,
            return_first=True
        )
        
        selected_restart = torch.argmax(acqf_values.mean(-1))
        return X[selected_restart].unsqueeze(0), hidden_state.detach()


    def get_next_X_in_topK(self, previous_Xy, previous_hidden_state):
        acqf_values, actions, hidden_states = self.forward(
            previous_Xy=previous_Xy,
            previous_hidden_state=previous_hidden_state,
            return_actions=True
        )
        
        selected_restart = torch.argmax(acqf_values.mean(-1))
        actions = actions[..., selected_restart, :, :].reshape(-1, self.parms.n_actions, self.parms.x_dim)
        random_in_topK = np.random.randint(0, high=actions.shape[0])
            
        random_action_idx = np.random.randint(0, high=self.parms.n_actions)
        X_topK = actions[random_in_topK, random_action_idx].reshape(1, self.parms.x_dim)
        
        hidden_state = hidden_states.reshape(-1, self.parms.n_restarts, self.parms.hidden_dim)[random_in_topK]
        hidden_state = hidden_state.reshape(self.parms.n_restarts, self.parms.hidden_dim)

        return X_topK.detach(), hidden_state.detach()
    
    
    def get_topK_actions(self, previous_Xy, previous_hidden_state):
        previous_Xy = previous_Xy.reshape(self.parms.n_restarts, self.parms.x_dim + self.parms.y_dim)
        
        # Compute Top-K
        _, hidden_state = self.get_optimization_variables(
            step=self.parms.lookahead_steps, Xy=previous_Xy, hidden_state=previous_hidden_state
        )
        
        actions = self.lin_map(hidden_state)
        # actions = torch.tanh(actions)
        new_shape = [self.parms.n_restarts, self.parms.n_actions, self.parms.x_dim]
        actions = actions.reshape(*new_shape)
        
        acqf_values = self.parms.compute_expectedloss_function(
            model=self.model, 
            actions=actions, 
            sampler=self.sampler, 
            info=self.parms
        )
        
        selected_restart = torch.argmax(acqf_values.mean(-1))
            
        topK_actions = actions[selected_restart].reshape(-1, self.parms.x_dim)
        topK_values = acqf_values[selected_restart].reshape(-1, 1)
        return topK_values, topK_actions