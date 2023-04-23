#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement an actor."""

import torch
from torch import Tensor
from typing import Dict
from botorch.acquisition import (
    qExpectedImprovement,
    qKnowledgeGradient,
    qMultiStepLookahead,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from models.amortized_network import AmortizedNetwork
from models.EHIG import qMultiStepEHIG, qLossFunctionTopK
from models.UncertaintySampling import qUncertaintySampling
from utils.plot import draw_losses


class Actor:
    r"""Actor class."""

    def __init__(self, parms, WM, buffer):
        """Initialize the actor.

        Args:
            parms (Parameters): A set of hyperparameters
            WM: World model
            buffer (Namespace): buffer must contain x and y
        """
        self.parms = parms
        if self.parms.use_amortized_optimization:
            self.maps = AmortizedNetwork(
                input_dim=self.parms.x_dim + self.parms.y_dim,
                output_dim=self.parms.x_dim,
                hidden_dim=self.parms.hidden_dim,
                n_actions=self.parms.n_actions,
                output_bounds=self.parms.bounds,
            ).double().to(self.parms.device)

            self._parameters = list(self.maps.parameters())
            self.prev_hid_state = torch.zeros(
                self.parms.n_restarts,
                self.parms.hidden_dim,
                device=self.parms.device,
                dtype=torch.double,
            )

        else:
            self.prev_hid_state = torch.empty(1)
            self.maps = []

        # Initialize some actor attributes
        self.cost = torch.zeros(1, device=self.parms.device).double()
        self.lookahead_steps = self.parms.lookahead_steps
        self.acqf = self.construct_acqf(WM=WM, buffer=buffer)

    def reset_parameters(self, seed=0, prev_X=None, prev_y=None):
        print("Resetting actor parameters...")
        torch.manual_seed(seed)

        if self.parms.use_amortized_optimization:
            optimizer = torch.optim.AdamW(self._parameters, lr=self.parms.acq_opt_lr)

            for _ in range(10):
                optimizer.zero_grad()
                return_dict = self.acqf.forward(
                    prev_X, prev_y, self.prev_hid_state, self.cost
                )

                X_randomized = (
                    torch.randn_like(return_dict["X"][0]) * 0.1
                ) + prev_X

                loss = torch.mean(
                    torch.pow(return_dict["X"][0] - X_randomized, 2)
                )  # MSE

                for i in range(1, self.lookahead_steps):
                    X_randomized = (
                        torch.randn_like(return_dict["X"][i]) * 0.1
                    ) + return_dict["X"][i - 1].detach()[None, ...].expand_as(
                        return_dict["X"][i]
                    )

                    loss += torch.mean(
                        torch.pow(return_dict["X"][i] - X_randomized, 2)
                    )  # MSE

                loss.backward()
                optimizer.step()

        else:
            self.maps = []

            for s in range(self.lookahead_steps):
                x = torch.rand(
                    self.parms.n_samples**s * self.parms.n_restarts,
                    self.parms.x_dim,
                    device=self.parms.device,
                ).double()
                self.maps.append((x * 2 - 1).requires_grad_(True))

            a = torch.rand(
                (self.parms.n_samples**self.lookahead_steps * self.parms.n_restarts)
                * self.parms.n_actions,
                self.parms.x_dim,
                device=self.parms.device,
            ).double()
            self.maps.append((a * 2 - 1).requires_grad_(True))
            self._parameters = self.maps

        if self.parms.algo == "HES":
            self.acqf.maps = self.maps

    def construct_acqf(self, WM, buffer=None):
        if self.parms.algo == "HES":
            return qMultiStepEHIG(
                model=WM,
                lookahead_steps=self.lookahead_steps,
                n_actions=self.parms.n_actions,
                n_fantasy_at_design_pts=[self.parms.n_samples] * self.lookahead_steps,
                n_fantasy_at_action_pts=self.parms.n_samples,
                maps=self.maps,
                loss_function_class=self.parms.loss_function_class,
                loss_function_hyperparameters=self.parms.loss_function_hyperparameters,
                cost_function_class=self.parms.cost_function_class,
                cost_function_hyperparameters=self.parms.cost_function_hyperparameters,
            )

        elif self.parms.algo == "us":
            return qUncertaintySampling(
                model=WM,
                parms=self.parms,
            )

        elif self.parms.algo == "kg":
            return qKnowledgeGradient(model=WM, num_fantasies=self.parms.n_samples)

        elif self.parms.algo == "qEI":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            return qExpectedImprovement(
                model=WM, best_f=buffer.y.max(), sampler=sampler
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
            return qUpperConfidenceBound(model=WM, beta=0.1, sampler=sampler)

        elif self.parms.algo == "qMSL":
            return qMultiStepLookahead(
                model=WM,
                batch_sizes=[1 for _ in range(self.lookahead_steps)],
                num_fantasies=[
                    self.parms.n_samples for _ in range(self.lookahead_steps)
                ],
            )

        else:
            raise ValueError(f"Unknown algo: {self.parms.algo}")

    def query(self, buffer, iteration: int):
        r"""Compute the next design point.

        Args:
            buffer: A ReplayBuffer object containing the data.
            iteration: The current iteration.
        """
        if self.acqf is None:
            data_x = self.uniform_random_sample_domain(self.parms.domain, 1)
            return data_x[0].reshape(1, -1)

        prev_X = (
            buffer.x[None, -1, :]
            .expand(self.parms.n_restarts, -1)
            .to(self.parms.device)
        )
        prev_y = (
            buffer.y[None, -1, :]
            .expand(self.parms.n_restarts, -1)
            .to(self.parms.device)
        )

        if abs(torch.rand(1).item()) > self.parms.epsilon:
            self.prev_hid_state = torch.rand(
                self.parms.n_restarts, self.parms.hidden_dim, device=self.parms.device
            ).double()
            prev_X = torch.rand_like(prev_y).double().to(self.parms.device)
            prev_y = torch.rand_like(prev_y).double().to(self.parms.device)

        # Reset the actor parameters for diversity
        self.reset_parameters(
            seed=iteration, prev_X=prev_X, prev_y=prev_y
        )

        # Optimize the acquisition function
        optimizer = torch.optim.AdamW(self._parameters, lr=self.parms.acq_opt_lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=1e-5
        )

        best_results = {}
        best_loss = float("inf")
        losses = []
        early_stop_counter = 0

        for opt_iter in range(self.parms.acq_opt_iter):
            optimizer.zero_grad()

            if self.parms.algo == "HES":
                return_dict = self.acqf.forward(
                    prev_X=prev_X,
                    prev_y=prev_y,
                    prev_hid_state=self.prev_hid_state,
                    previous_cost=self.cost,
                )
                loss = -return_dict["acqf_values"].sum()
            else:
                X = torch.cat(self.maps, dim=0)
                acqf_values = self.acqf.forward(X=X)
                loss = -acqf_values.sum()

            losses.append(loss.item())
            print("Loss:", loss.item())

            # Get the best results
            if opt_iter >= self.parms.acq_warmup_iter:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    # We need to follow the plan step by step
                    if self.parms.algo == "HES":
                        best_results = self.get_next_X_and_optimal_actions(return_dict)
                    else:
                        best_results = {}
                        X = torch.cat(self.maps, dim=0)
                        best_results["next_X"] = (
                            self.acqf.get_multi_step_tree_input_representation(X)[0]
                            .cpu()
                            .detach()
                        )
                        best_results["acqf_values"] = acqf_values.cpu().detach()
                        best_results["optimal_actions"] = self.maps[-1].cpu().detach()
                        best_results["hidden_state"] = self.prev_hid_state

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
        self.prev_hid_state = best_results["hidden_state"]
        self.prev_hid_state = self.prev_hid_state.to(self.parms.device)

        # Update new cost
        # self.cost = best_results["cost"]

        # Draw losses by acq_opt_iter using matplotlib
        draw_losses(config=self.parms, losses=losses, iteration=iteration)

        return (
            best_results["next_X"],
            best_results["optimal_actions"],
            best_results["acqf_values"],
        )

    def get_next_X_and_optimal_actions(self, return_dict: Dict[str, Tensor]):
        """Choose the index of restart that has maximal acqf_value.

        Args:
            return_dict (dict): Return dictionary from acqf forward.

        Returns:
            dict: Dictionary of next_X, acqf_values, hidden_state,
                  optimal_actions and selected_restart.
        """
        chosen_idx = torch.argmax(return_dict["acqf_values"])
        next_X = return_dict["X"][0][chosen_idx].reshape(1, -1)

        hidden_state = return_dict["hidden_state"][0]
        if self.parms.use_amortized_optimization:
            hidden_state = hidden_state[chosen_idx: chosen_idx + 1]
            hidden_state = hidden_state.expand_as(
                return_dict["hidden_state"][0]
            )

        topK_actions = return_dict["actions"][..., chosen_idx, :, :].reshape(
            -1, *return_dict["actions"].shape[-2:]
        )
        topK_values = return_dict["acqf_values"][chosen_idx].reshape(-1, 1)

        return {
            "next_X": next_X.detach().cpu(),
            "acqf_values": topK_values.detach().cpu(),
            "hidden_state": hidden_state.detach().cpu(),
            "optimal_actions": topK_actions.detach().cpu(),
            "selected_restart": chosen_idx.detach().cpu(),
        }
