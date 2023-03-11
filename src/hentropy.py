#!/usr/bin/env python
"""Non-myopic H-entropy search"""

from src.variance_reduction import baseline
from src.nn import MLP
from src.tasks import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from typing import Optional

__author__ = ""
__copyright__ = "Copyright 2022, Stanford University"

import torch
import math

patch_typeguard()


class HEntropySearch(MCAcquisitionFunction):
    @typechecked
    def __init__(
        self,
        config,
        model: Model,
        data,
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
        self.data = data
        self.set_X_pending(batch_as_pending)
        self.baseline_initialization = True
        self.p_yi_xiDi = {}
        self.p_f_Di = {}
        self.p_f_Di[0] = self.model
        self.yis = {}

        # value function
        if c.app == "topk":
            self.HEntropySearch_Task = HEntropySearchTopK
        elif c.app == "minmax":
            self.HEntropySearch_Task = HEntropySearchMinMax
        elif c.app == "twoval":
            self.HEntropySearch_Task = HEntropySearchTwoVal
        elif c.app == "mvs":
            self.HEntropySearch_Task = HEntropySearchMVS
        elif c.app == "levelset":
            self.HEntropySearch_Task = HEntropySearchLevelSet
        elif c.app == "multilevelset":
            self.HEntropySearch_Task = HEntropySearchMultiLevelSet
        elif c.app == "pbest":
            self.HEntropySearch_Task = HEntropySearchPbest
        elif c.app == "bestofk":
            self.HEntropySearch_Task = HEntropySearchBestOfK
        else:
            raise NotImplemented

        # base samples should be fixed for joint optimization over batch_a1s
        self.samplers = {}
        for i in range(c.lookahead_steps + 1):
            num_samples = math.ceil(c.n_samples / (c.decay_factor ** i))
            self.samplers[i] = SobolQMCNormalSampler(
                num_samples=num_samples, resample=False, collapse_batch_dims=True
            )

        if c.baseline:
            n_dim_input_h = (c.init_data.x.shape[0] + c.lookahead_steps) * (
                1 + c.n_dim_design
            )
            self.maps_h = [
                MLP(
                    input_size=n_dim_input_h,
                    n_neurons=math.ceil(n_dim_input_h * c.hidden_coeff),
                    n_layers=c.n_layers,
                    output_size=1,
                    activation=c.activation,
                    last_layer_linear=True,
                )
                .to(c.device)
                .double()
                for _ in range(c.n_restarts)
            ]
            c.baseline = baseline(c, self.po, self.model,
                                  self.sampler, self.yis)

        if not c.vi:
            self.po = self.directly_parameterize_output()
            self.po[-1] = data.x[-1]
            for i in range(c.lookahead_steps):
                self.sample_yi(i)

        else:
            self.maps_i = {}
            for i in range(c.lookahead_steps + 1):
                dim = c.n_dim_action if i == c.lookahead_steps else c.n_dim_design
                input_size = 1 if i == 0 else 1 + (i - 1) * (1 + dim)
                self.maps_i[i] = [
                    MLP(
                        input_size=input_size,
                        n_neurons=math.ceil(input_size * c.hidden_coeff),
                        n_layers=c.n_layers,
                        output_size=dim,
                        activation=c.activation,
                        last_layer_linear=True,
                    )
                    .to(c.device)
                    .double()
                    for _ in range(c.n_restarts)
                ]

            self.inputs = []
            self.po = {}
            self.po[-1] = data.x[-1]
            for i in range(c.lookahead_steps + 1):
                inputs_i = []
                if i == 0:
                    inputs_i = torch.ones(
                        [c.n_restarts, 1], device=c.device).double()
                    self.inputs.append(inputs_i)
                    self.po[i] = [
                        (self.maps_i[i][j](self.inputs[i][j, :]))[..., None, :]
                        for j in range(self.config.n_restarts)
                    ]

                else:
                    if i == 1:
                        self.sample_yi(0)
                        inputs_i = self.yis[0]
                        self.inputs.append(inputs_i)

                    else:
                        self.sample_yi(i - 1)
                        for j in range(i - 1):
                            new_dim = math.ceil(
                                c.n_samples / (c.decay_factor ** (j + 1))
                            )
                            y_j_x_jplus1_pair = torch.cat(
                                [self.yis[j], self.po[j + 1].clone()], dim=-1
                            )
                            num_expand_dim = len(self.yis[i - 1].shape) - len(
                                y_j_x_jplus1_pair.shape
                            )
                            org_shape = y_j_x_jplus1_pair.shape

                            y_j_x_jplus1_pair = y_j_x_jplus1_pair.view(
                                *[1] * num_expand_dim, *org_shape
                            )

                            dim_expander_matrix = torch.ones(
                                [*[new_dim] * num_expand_dim,
                                    *[1] * len(org_shape)],
                                device=c.device,
                            )

                            y_j_x_jplus1_pair = dim_expander_matrix * (
                                y_j_x_jplus1_pair
                            )

                            inputs_i.append(y_j_x_jplus1_pair)

                        inputs_i.append(self.yis[i - 1])
                        inputs_i = torch.cat(inputs_i, dim=-1)
                        self.inputs.append(inputs_i)

                    self.po[i] = [
                        self.maps_i[i][j](self.inputs[i][..., j, :, :])
                        for j in range(self.config.n_restarts)
                    ]

                self.po[i] = torch.stack(self.po[i], dim=-3)
            self.init_nn()

    @typechecked
    def forward(self) -> TensorType["n_restarts"]:
        c = self.config

        if c.vi:
            for i in range(c.lookahead_steps + 1):
                if i == 0:
                    self.po[i] = [
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
                                [self.yis[j], self.po[j + 1]], dim=-1
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

                    self.po[i] = [
                        self.maps_i[i][j](self.inputs[i][..., j, :, :])
                        for j in range(self.config.n_restarts)
                    ]

                self.po[i] = torch.stack(self.po[i], dim=-3)

        self.hes_task = self.HEntropySearch_Task(
            config=c,
            model=self.p_f_Di[c.lookahead_steps],
            sampler=self.samplers[c.lookahead_steps],
        )

        return self.hes_task(po=self.po)

    @typechecked
    def sample_yi(self, i) -> None:
        with torch.no_grad():
            self.p_yi_xiDi[i] = self.p_f_Di[i].posterior(X=self.po[i])
            self.yis[i] = self.samplers[i](self.p_yi_xiDi[i])

            self.p_f_Di[i + 1] = self.p_f_Di[i].fantasize(
                X=self.po[i],
                sampler=self.samplers[i],
            )

    @typechecked
    def directly_parameterize_output(self):
        """init and return batch output tensors, where output can be
        either action or design."""

        c = self.config
        po = {}

        for i in range(c.lookahead_steps + 1):
            dim = c.n_dim_design if i == c.lookahead_steps else c.n_dim_design
            n = c.n_actions if i == c.lookahead_steps else c.n_candidates
            if i == 0:
                output_dim = [c.n_restarts, n, dim]
            else:
                output_dim = [c.n_restarts, n, dim]
                for j in range(i):
                    output_dim.insert(
                        0, math.ceil(c.n_samples // (c.decay_factor ** j))
                    )

            b = c.bounds_action if i == c.lookahead_steps else c.bounds_design
            noise = b[0] + (b[1] - b[0]) * \
                torch.rand(output_dim, device=c.device)

            if c.r and c.local_init:
                previous_output = self.data.x[-1].detach().expand(*output_dim)
                po[i] = previous_output + 0.1 * c.r * noise
            else:  # uniform initialization
                po[i] = noise

            po[i].requires_grad_(True)
        return po

    # neural model initialization
    @typechecked
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
