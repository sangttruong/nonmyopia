#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement an actor."""

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from botorch.acquisition import (
    qExpectedImprovement,
    qKnowledgeGradient,
    qMultiStepLookahead,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)

# from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf
from _3_amortized_network import AmortizedNetwork, Project2Range
from _4_qhes import qMultiStepHEntropySearch
from _5_evalplot import draw_loss_and_cost
from _6_samplers import DesireSobolQMCNormalSampler as SobolQMCNormalSampler


class Actor:
    r"""Actor class."""

    def __init__(self, parms):
        """Initialize the actor.

        Args:
            parms (Parameters): A set of hyperparameters
        """
        self.parms = parms
        if self.parms.algo == "HES":
            if self.parms.amortized:
                self.maps = AmortizedNetwork(
                    input_dim=self.parms.x_dim + self.parms.y_dim,
                    output_dim=self.parms.x_dim,
                    hidden_dim=self.parms.hidden_dim,
                    n_actions=self.parms.n_actions,
                    output_bounds=self.parms.bounds,
                )
                self.maps = self.maps.to(
                    dtype=self.parms.torch_dtype, device=self.parms.device
                )

                self._parameters = list(self.maps.parameters())

            else:
                self.maps = []

        # Initialize some actor attributes
        self.lookahead_steps = self.parms.lookahead_steps

    def reset_parameters(
        self,
        prev_X: Tensor,
        prev_y: Tensor,
        prev_hid_state: Tensor,
    ):
        r"""Reset actor parameters.

        With amortized version, this function optimizes the
        output X of acquisition function to randomized X.
        While in the non-amortized version, this function
        just initializes random X.

        Args:
            prev_X (Tensor): Previous design points
            prev_y (Tensor): Previous observations
        """
        project2range = Project2Range(self.parms.bounds[0], self.parms.bounds[1])

        if self.parms.amortized:
            optimizer = torch.optim.AdamW(self._parameters, lr=self.parms.acq_opt_lr)

            for _ in range(10):
                optimizer.zero_grad()
                return_dict = self.acqf(
                    prev_X=prev_X,
                    prev_y=prev_y,
                    prev_hid_state=prev_hid_state,
                    maps=self.maps,
                )

                loss = 0
                for i in range(self.lookahead_steps):
                    X_randomized = torch.rand_like(return_dict["X"][i])
                    # min max scaling
                    X_randomized = (
                        X_randomized * (self.parms.bounds[1] - self.parms.bounds[0])
                        + self.parms.bounds[0]
                    )
                    loss += (return_dict["X"][i] - X_randomized).pow(2).mean()

                loss.backward()
                optimizer.step()

        else:
            self.maps = []

            for s in range(self.lookahead_steps):
                x = torch.rand(
                    self.parms.n_samples**s * self.parms.n_restarts,
                    self.parms.x_dim,
                    device=self.parms.device,
                    dtype=self.parms.torch_dtype,
                )
                x = project2range(x).requires_grad_(True)
                self.maps.append(x)

            a = torch.rand(
                self.parms.n_samples**self.lookahead_steps
                * self.parms.n_restarts
                * self.parms.n_actions,
                self.parms.x_dim,
                device=self.parms.device,
                dtype=self.parms.torch_dtype,
            )
            a = project2range(a).requires_grad_(True)
            self.maps.append(a)
            self._parameters = self.maps

    def construct_acqf(self, WM, buffer):
        """Contruct aquisition function.

        Args:
            WM: World model.
            buffer: A ReplayBuffer object containing the data.

        Raises:
            ValueError: If defined algo is not implemented.

        Returns:
            AcquisitionFunction: An aquisition function instance
        """
        if self.parms.algo == "HES":
            nf_design_pts = [self.parms.n_samples] * 4
            nf_design_pts = nf_design_pts + [1] * (self.lookahead_steps - 4)

            self.acqf = qMultiStepHEntropySearch(
                model=WM,
                lookahead_steps=self.lookahead_steps,
                n_actions=self.parms.n_actions,
                n_fantasy_at_design_pts=nf_design_pts,
                n_fantasy_at_action_pts=self.parms.n_samples,
                loss_function_class=self.parms.loss_function_class,
                loss_func_hypers=self.parms.loss_func_hypers,
                cost_function_class=self.parms.cost_function_class,
                cost_func_hypers=self.parms.cost_func_hypers,
            )

        elif self.parms.algo == "kg":
            self.acqf = qKnowledgeGradient(model=WM, num_fantasies=self.parms.n_samples)

        elif self.parms.algo == "qEI":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            self.acqf = qExpectedImprovement(
                model=WM, best_f=buffer["y"].max(), sampler=sampler
            )

        elif self.parms.algo == "qPI":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            self.acqf = qProbabilityOfImprovement(
                model=WM, best_f=buffer["y"].max(), sampler=sampler
            )

        elif self.parms.algo == "qSR":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            self.acqf = qSimpleRegret(model=WM, sampler=sampler)

        elif self.parms.algo == "qUCB":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            self.acqf = qUpperConfidenceBound(model=WM, beta=0.1, sampler=sampler)

        elif self.parms.algo == "qMSL":
            num_fantasies = [self.parms.n_samples] * self.lookahead_steps
            self.acqf = qMultiStepLookahead(
                model=WM,
                batch_sizes=[1] * self.lookahead_steps,
                num_fantasies=num_fantasies,
            )

        else:
            raise ValueError(f"Unknown algo: {self.parms.algo}")

    def query(self, buffer, iteration: int):
        r"""Compute the next design point.

        Args:
            prev_X (Tensor): Previous design points.
            prev_y (Tensor): Previous observations.
            prev_hid_state (Tensor): Previous hidden state.
            iteration: The current iteration.
        """
        if self.acqf is None:
            data_x = self.uniform_random_sample_domain(self.parms.domain, 1)
            return data_x[0].reshape(1, -1)

        prev_X = buffer["x"][iteration - 1 : iteration].expand(
            self.parms.n_restarts, -1
        )
        prev_y = buffer["y"][iteration - 1 : iteration].expand(
            self.parms.n_restarts, -1
        )
        prev_hid_state = buffer["h"][iteration - 1 : iteration].expand(
            self.parms.n_restarts, -1
        )

        # Optimize the acquisition function
        if self.parms.algo == "HES":
            # Reset the actor parameters for diversity
            if iteration == self.parms.n_initial_points:
                self.reset_parameters(
                    prev_X=prev_X, prev_y=prev_y, prev_hid_state=prev_hid_state
                )

            optimizer = torch.optim.AdamW(self._parameters, lr=self.parms.acq_opt_lr)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=20, eta_min=1e-5
            )
            losses = []
            costs = []

            for ep in range(self.parms.acq_opt_iter):
                print(f"\nEpoch {ep:05d}", end="\t", flush=True)
                return_dict = self.acqf.forward(
                    prev_X=prev_X,
                    prev_y=prev_y,
                    prev_hid_state=prev_hid_state,
                    maps=self.maps,
                )

                acqf_loss = return_dict["acqf_loss"].mean()
                acqf_cost = return_dict["acqf_cost"].mean()
                # >> n_restart

                losses.append(acqf_loss.item())
                costs.append(acqf_cost.item())

                loss = acqf_loss + acqf_cost
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                print(f"Loss {loss.item():.5f}", end="\r")

            loss = return_dict["acqf_loss"] + return_dict["acqf_cost"]

            # Choose which restart produce the lowest loss
            idx = torch.argmin(loss)

            # Get next X as X_0 at idx
            next_X = return_dict["X"][0][idx].reshape(1, -1)
            # >>> 1 * x_dim

            # Get next hidden state of X_0 at idx
            hidden_state = return_dict["hidden_state"][0]
            # >>> n_restarts * hidden_dim

            if self.parms.amortized:
                hidden_state = hidden_state[idx : idx + 1]
                # >>> n_restarts * hidden_dim

            acqf_loss = loss[idx]
            print("Acqf loss:", acqf_loss.item())
            # >>> n_actions * 1

            actions = return_dict["actions"][..., idx, :, :]

            # Draw losses by acq_opt_iter
            draw_loss_and_cost(self.parms.save_dir, losses, costs, iteration)

        else:
            bounds = torch.tensor(
                [self.parms.bounds] * self.parms.x_dim,
                dtype=self.parms.torch_dtype,
                device=self.parms.device,
            ).T
            # Optimize acqf
            next_X, acqf_loss = optimize_acqf(
                acq_function=self.acqf,
                bounds=bounds,
                q=self.parms.n_actions,
                num_restarts=self.parms.n_restarts,
                raw_samples=self.parms.n_samples,
            )

            hidden_state = prev_hid_state[0]
            actions = None

        next_X = next_X.detach()
        actions = actions.detach()
        acqf_loss = acqf_loss.detach()
        hidden_state = hidden_state.detach()

        return next_X, hidden_state, actions
