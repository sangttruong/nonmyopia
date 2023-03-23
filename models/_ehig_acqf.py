import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from typing import Optional
import torch
import math
import numpy as np
import copy

class EHIG(MCAcquisitionFunction):
    def __init__(
        self,
        config,
        compute_expectedloss_function,
        model: Model,
        batch_as_pending: Optional[torch.Tensor] = None,
    ) -> None:
        r"""
        Args:
            config
            model: A fitted model. Must support fantasizing.
            samplers: A list of samplers used to sample fantasy observations. Optional
                if `n_samples` is specified.
            batch_as_pending
        """
        super(MCAcquisitionFunction, self).__init__(model=model)

        self.config = c = config
        self.set_X_pending(batch_as_pending)
        self.baseline_initialization = True
        self.p_yi_xiDi = {}
        self.p_f_Di = {}
        self.p_f_Di[0] = self.model
        self.yis = {}
        self.data = None

        # H-entropy loss function
        self.compute_expectedloss_function = compute_expectedloss_function

        # base samples should be fixed for joint optimization over batch_a1s
        self.samplers = {}
        for i in range(c.lookahead_steps + 1):
            num_samples = math.ceil(c.n_samples / (c.decay_factor ** i))
            self.samplers[i] = SobolQMCNormalSampler(
                sample_shape=num_samples, resample=False, collapse_batch_dims=True
            )

        # parameterize optimization variables
        # TODO_DUC: set optimization following this example so that later we don't 
        # have to manually provide the parameters to optimizer
        # self.weight = nn.Parameter(torch.randn(num_features))
        # depending on whether we use "amortized_optimization" or not (VI vs MC)
        # the parameters are tensors (x0, x1, ...) or neural network (e.g. RNN)
        # lookahead_steps
        # TODO: need to know the shape of y before parameterizing x!!! 
        # probably a better idea to parameterize on the fly using
        # compute_optimization_variable
        
        if self.use_amortized_optimization:
            # ----------------- Pseudo code -----------------
            # create an RNN with input size of dim(x) + dim(y) and output size of dim(x)
            # TODO: Infer i dimension
            y_dim = 1
            self.amortized_maps = torch.nn.Sequential(
                nn.Linear(self.parms.n_dim + y_dim, self.parms.n_dim), 
                nn.ReLU(),
                nn.GRUCell(self.parms.n_dim, self.parms.n_dim)
            )
            # self.init_nn()

        else:
            self.amortized_maps = dict()

    def compute_optimization_variable(self, y):
        if self.use_amortized_optimization:
            return self.amortized_map(y)

        else:
            pass
            # time_step = y.shape[0]
            # if time_step not in self.amortized_map.keys():
            #     self.amortized_map[time_step] = nn.Parameter(torch.randn( ... some correct shape ... ))
            # return self.amortized_map[time_step]
            
            
    def forward(self):
        c = self.config

        if c.vi:
            data_tensor = torch.cat([self.data.x, self.data.y], dim=-1)
            
            for i in range(c.lookahead_steps + 1):
                next_x = self.amortized_maps(data_tensor)[-1]
                next_y = self.model(next_x) # There might be multiple y
                
                # Join (x,y) to data
                data_tensor = torch.cat([
                                            data_tensor, 
                                            torch.cat([next_x, next_y], dim=-1)
                                        ], dim=0)
                
                
                # if i == 0:
                #     self.po[i] = [
                #         (self.maps_i[i][j](self.inputs[i][j, :]))[..., None, :]
                #         for j in range(self.config.n_restarts)
                #     ]

                # else:
                #     if i > 1:
                #         inputs_i = []
                #         for j in range(i - 1):
                #             new_dim = math.ceil(
                #                 c.n_samples / (c.decay_factor ** (j + 1))
                #             )
                #             y_j_x_jplus1_pair = torch.cat(
                #                 [self.yis[j], self.po[j + 1]], dim=-1
                #             )
                #             num_expand_dim = len(self.yis[i - 1].shape) - len(
                #                 y_j_x_jplus1_pair.shape
                #             )
                #             org_shape = y_j_x_jplus1_pair.shape
                #             y_j_x_jplus1_pair = y_j_x_jplus1_pair.view(
                #                 *[1] * num_expand_dim, *org_shape
                #             )
                #             y_j_x_jplus1_pair = y_j_x_jplus1_pair.expand(
                #                 *[new_dim] * num_expand_dim, *org_shape
                #             )
                #             inputs_i.append(y_j_x_jplus1_pair)
                #         inputs_i.append(self.yis[i - 1])
                #         inputs_i = torch.cat(inputs_i, dim=-1)
                #         self.inputs[i] = inputs_i

                #     self.po[i] = [
                #         self.maps_i[i][j](self.inputs[i][..., j, :, :])
                #         for j in range(self.config.n_restarts)
                #     ]

                # self.po[i] = torch.stack(self.po[i], dim=-3)

        else:
            po = self.directly_parameterize_output()
            po[-1] = self.data.x[-1]
            for i in range(c.lookahead_steps):
                self.sample_yi(i)
        
        self.hes_task = self.compute_expectedloss_function(
            config=c,
            model=self.p_f_Di[c.lookahead_steps],
            sampler=self.samplers[c.lookahead_steps],
        )

        total_cost = 0
        if c.r:
            threshold1 = torch.nn.Threshold(-c.r, 100)
            threshold2 = torch.nn.Threshold(0, 0)
            for i in range(c.lookahead_steps):
                distance = ((po[i] - po[i - 1]) **
                            2).sum((-1, -2), keepdim=True)
                cost = threshold2(threshold1(-distance))
                total_cost = total_cost + cost

        
        return self.hes_task(po=po)

    def directly_parameterize_output_topk(self, data):
        """[WIP] init and return batch_x0s, batch_a1s tensors for topk."""
        mc_params = self.directly_parameterize_output(data)

        batch_x0s = mc_params[0]
        batch_a1s = mc_params[self.config.lookahead_steps]

        # init actions to topk diverse data points
        config = self.config  # for brevity
        data_y = copy.deepcopy(np.array(data.y).reshape(-1))
        data_x = copy.deepcopy(
            np.array(data.x.cpu()).reshape(-1, config.n_dim_design))
        for i in range(config.n_actions):
            if len(data_y) > 0:
                topk_idx = data_y.argmax()
                topk_x = data_x[topk_idx]
                dists = np.linalg.norm(data_x - topk_x, axis=1, ord=1)
                del_idx = np.where(dists < config.dist_threshold)[0]
                data_x = np.delete(data_x, del_idx, axis=0)
                data_y = np.delete(data_y, del_idx, axis=0)
                print(f"topk_x {i} = {topk_x}")
                with torch.no_grad():
                    batch_a1s[:, :, i, :] = torch.tensor(topk_x)
            else:
                pass

        return batch_x0s, batch_a1s.to(self.config.device)
    
    def sample_yi(self, i) -> None:
        with torch.no_grad():
            self.p_yi_xiDi[i] = self.p_f_Di[i].posterior(X=self.po[i])
            self.yis[i] = self.samplers[i](self.p_yi_xiDi[i])
            
            self.p_f_Di[i + 1] = self.p_f_Di[i].fantasize(
                X=self.po[i],
                sampler=self.samplers[i],
            )

    def directly_parameterize_output(self):
        """init and return batch output tensors, where output can be
        either action or design."""

        c = self.config
        po = {}

        for i in range(c.lookahead_steps + 1):
            dim = c.n_dim if i == c.lookahead_steps else c.n_dim
            n = c.n_actions if i == c.lookahead_steps else c.n_candidates
            if i == 0:
                output_dim = [c.n_restarts, n, dim]
            else:
                output_dim = [c.n_restarts, n, dim]
                for j in range(i):
                    output_dim.insert(
                        0, math.ceil(c.n_samples // (c.decay_factor ** j))
                    )

            b = c.bounds if i == c.lookahead_steps else c.bounds
            noise = b[0] + (b[1] - b[0]) * \
                torch.rand(output_dim, device=c.device, dtype=c.torch_dtype)

            if c.r and c.local_init:
                previous_output = self.data.x[-1].detach().expand(*output_dim)
                po[i] = previous_output + 0.1 * c.r * noise
            else:  # uniform initialization
                po[i] = noise

            po[i].requires_grad_(True)
        return po

    def init_nn(self) -> None:
        print("Initialize neural network")
        c = self.config

        params = []
        params_ = [
            self.maps_i[i][j].parameters()
            for j in range(c.n_restarts)
            for i in range(c.lookahead_steps + 1)
        ]
        for i in range(len(params_)):
            params += list(params_[i])

        optim = torch.optim.Adam(params, lr=c.acq_opt_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=c.T_max, eta_min=c.eta_min
        )

        losses = []
        lrs = []
        patient = c.max_patient
        min_loss = float("inf")
        dpo = self.directly_parameterize_output()
        po = {}
        for iteration in tqdm(range(c.acq_opt_iter)):

            loss = 0
            for i in range(c.lookahead_steps + 1):
                if i == 0:
                    po[i] = [
                        (self.maps_i[i][j](self.inputs[i][j, :]))[..., None, :]
                        for j in range(self.config.n_restarts)
                    ]

                else:
                    if i > 1:
                        inputs_i = []
                        for j in range(i - 1):
                            new_dim = math.ceil(
                                c.n_samples / (c.decay_factor ** (j + 1))
                            )
                            y_j_x_jplus1_pair = torch.cat(
                                [self.yis[j], po[j + 1]], dim=-1
                            )
                            num_expand_dim = len(self.yis[i - 1].shape) - len(
                                y_j_x_jplus1_pair.shape
                            )
                            org_shape = y_j_x_jplus1_pair.shape
                            y_j_x_jplus1_pair = y_j_x_jplus1_pair.view(
                                *[1] * num_expand_dim, *org_shape
                            )
                            y_j_x_jplus1_pair = y_j_x_jplus1_pair.expand(
                                *[new_dim] * num_expand_dim, *org_shape
                            )
                            inputs_i.append(y_j_x_jplus1_pair)

                        inputs_i.append(self.yis[i - 1])
                        inputs_i = torch.cat(inputs_i, dim=-1)
                        self.inputs[i] = inputs_i

                    po[i] = [
                        self.maps_i[i][j](self.inputs[i][..., j, :, :])
                        for j in range(self.config.n_restarts)
                    ]

                po[i] = torch.stack(po[i], dim=-3)

                loss = loss + ((dpo[i] - po[i]) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            losses.append(loss.item())
            lrs.append(scheduler.get_last_lr())
            if loss < min_loss:
                min_loss = loss
                patient = c.max_patient
            else:
                patient -= 1

            if patient < 0:
                break

        print(f"Initialize result: loss {loss.item()} at epoch {iteration} ")

        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(np.array(losses)[100:], "b-", linewidth=1)
        ax2.plot(np.array(lrs)[100:], "r-", linewidth=1)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color="b")
        ax2.set_ylabel("Learning rate", color="r")
        plt.savefig(f"{c.save_dir}/init_optim_{i}.png", bbox_inches="tight")
