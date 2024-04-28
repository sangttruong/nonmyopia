#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement an actor."""

import torch
from torch import Tensor
from tqdm import tqdm
from botorch.acquisition import (
    qExpectedImprovement,
    qKnowledgeGradient,
    qMultiStepLookahead,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
    qNegIntegratedPosteriorVariance,
)

from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf, optimize_acqf_discrete
from _3_amortized_network import AmortizedNetwork, Project2Range
from _3_amortized_network_antbo import AmortizedNetworkAntBO
from _4_qhes import qMultiStepHEntropySearch
from _4_qhes_ts import qMultiStepHEntropySearchTS
from _5_evalplot import draw_loss_and_cost
from _9_semifuncs import nm_AAs
from _10_budgeted_bo import (
    BudgetedMultiStepExpectedImprovement,
    get_suggested_budget,
    optimize_acqf_and_get_suggested_point,
    evaluate_obj_and_cost_at_X,
)


class Actor:
    r"""Actor class."""

    def __init__(self, parms):
        """Initialize the actor.

        Args:
            parms (Parameters): A set of hyperparameters
        """
        self.parms = parms
        if self.parms.algo_ts:
            self.acqf_class = qMultiStepHEntropySearchTS
        else:
            self.acqf_class = qMultiStepHEntropySearch

        if self.parms.algo == "HES":
            if self.parms.amortized:
                if self.parms.env_name == "AntBO":
                    amor_net = AmortizedNetworkAntBO
                else:
                    amor_net = AmortizedNetwork

                self.maps = amor_net(
                    input_dim=self.parms.x_dim + self.parms.y_dim,
                    output_dim=self.parms.x_dim,
                    hidden_dim=self.parms.hidden_dim,
                    n_actions=self.parms.n_actions,
                    output_bounds=self.parms.bounds,
                    discrete=parms.env_discretized,
                    num_categories=self.parms.num_categories,
                )
                self.maps = self.maps.to(
                    dtype=self.parms.torch_dtype, device=self.parms.device
                )
                print(
                    "Number of AmortizedNet params:",
                    sum(p.numel()
                        for p in self.maps.parameters() if p.requires_grad),
                )

                self._parameters = list(self.maps.parameters())

            else:
                self.maps = []

        # Initialize some actor attributes
        self.algo_lookahead_steps = self.parms.algo_lookahead_steps
        self.acqf_params = {}

    def reset_parameters(
        self,
        prev_X: Tensor,
        prev_y: Tensor,
        prev_hid_state: Tensor,
        embedder=None,
    ):
        r"""Reset actor parameters.

        With amortized version, this function optimizes the
        output X of acquisition function to randomized X.
        While in the non-amortized version, this function
        just initializes random X.

        Args:
            prev_X (Tensor): Previous design points - Decoded in case of discrete
            prev_y (Tensor): Previous observations
        """
        print("Resetting actor parameters...")
        project2range = Project2Range(
            self.parms.bounds[..., 0], self.parms.bounds[..., 1]
        )

        if self.parms.amortized:
            optimizer = torch.optim.AdamW(
                self._parameters, lr=self.parms.acq_opt_lr)

            if embedder is not None:
                encoded_prev_X = embedder.encode(prev_X)
            else:
                encoded_prev_X = prev_X

            A_randomized = torch.rand(
                (1000, 2), device=self.parms.device, dtype=self.parms.torch_dtype
            )
            ub = torch.clamp(
                encoded_prev_X[0]
                + self.parms.cost_func_hypers["radius"]
                * (self.algo_lookahead_steps + 1),
                max=self.parms.bounds[..., 1],
            )
            lb = torch.clamp(
                encoded_prev_X[0]
                - self.parms.cost_func_hypers["radius"]
                * (self.algo_lookahead_steps + 1),
                min=self.parms.bounds[..., 0],
            )
            A_randomized = (A_randomized * (ub - lb) + lb).detach()

            X_randomizeds = []
            for i in range(self.algo_lookahead_steps):
                X_randomized = torch.rand(
                    (1000, 2), device=self.parms.device, dtype=self.parms.torch_dtype
                )
                ub = torch.clamp(
                    encoded_prev_X[0] +
                    self.parms.cost_func_hypers["radius"] * (i + 1),
                    max=self.parms.bounds[..., 1],
                )
                lb = torch.clamp(
                    encoded_prev_X[0] -
                    self.parms.cost_func_hypers["radius"] * (i + 1),
                    min=self.parms.bounds[..., 0],
                )

                X_randomized = (X_randomized * (ub - lb) + lb).detach()
                X_randomizeds.append(X_randomized)

            std = (self.parms.bounds[..., 1] -
                   self.parms.bounds[..., 0]).mean() / 2

            for _ in tqdm(range(20)):
                optimizer.zero_grad()
                return_dict = self.acqf(
                    prev_X=prev_X,
                    prev_y=prev_y,
                    prev_hid_state=prev_hid_state,
                    maps=self.maps,
                    embedder=embedder,
                )

                loss = 0
                for i in range(self.algo_lookahead_steps):
                    mean = return_dict["X"][i].mean(
                        dim=tuple(range(return_dict["X"][i].dim() - 1))
                    )
                    dist = torch.distributions.Normal(mean, std)
                    loss += -dist.log_prob(X_randomizeds[i]).mean()

                mean = return_dict["actions"].mean(
                    dim=tuple(range(return_dict["actions"].dim() - 1))
                )
                dist = torch.distributions.Normal(mean, std)
                loss += -dist.log_prob(A_randomized).mean()

                print(f"Loss resetting: {loss.item():.5f}",
                      end="\r", flush=True)

                grads = torch.autograd.grad(
                    loss, self._parameters, allow_unused=True)
                for param, grad in zip(self._parameters, grads):
                    param.grad = grad
                optimizer.step()

        else:
            self.maps = []

            for s in range(self.algo_lookahead_steps):
                x = torch.rand(
                    self.parms.n_samples**s * self.parms.n_restarts,
                    self.parms.x_dim,
                    device=self.parms.device,
                    dtype=self.parms.torch_dtype,
                )
                x = project2range(x).requires_grad_(True)
                self.maps.append(x)

            a = torch.rand(
                self.parms.n_samples**self.algo_lookahead_steps
                * self.parms.n_restarts
                * self.parms.n_actions,
                self.parms.x_dim,
                device=self.parms.device,
                dtype=self.parms.torch_dtype,
            )
            a = project2range(a).requires_grad_(True)
            self.maps.append(a)
            self._parameters = self.maps

    def set_initial_acqf_params(self):
        """Set initial acquisition function parameters."""
        if self.parms.algo == "BudgetedBO":
            self.objective_X = torch.empty(0)
            self.cost_X = torch.empty(0)
            self.acqf_params["initial_budget"] = self.parms.budget
            self.acqf_params["current_budget"] = None
            self.acqf_params["current_budget_plus_cumulative_cost"] = None
            self.acqf_params["lower_bound"] = None
            self.acqf_params["budget_left"] = self.parms.budget

    def construct_acqf(self, WM, buffer, **kwargs):
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
            if self.parms.algo_ts:
                nf_design_pts = [1] * self.algo_lookahead_steps
            else:
                if self.algo_lookahead_steps == 1:
                    nf_design_pts = [64]
                elif self.algo_lookahead_steps == 2:
                    nf_design_pts = [64, 8]  # [64, 64]
                elif self.algo_lookahead_steps == 3:
                    nf_design_pts = [64, 4, 2]  # [64, 32, 8]
                elif self.algo_lookahead_steps >= 4:
                    nf_design_pts = [64, 4, 2, 1]  # [16, 8, 8, 8]
                    nf_design_pts = nf_design_pts + [1] * (
                        self.algo_lookahead_steps - 4
                    )

                # nf_design_pts = [self.parms.n_samples]
                # for s in range(1, self.algo_lookahead_steps):
                #     next_nf = nf_design_pts[s-1] // 4
                #     if next_nf < 1:
                #         next_nf = 1
                #     nf_design_pts.append(next_nf)

            self.acqf = self.acqf_class(
                model=WM,
                lookahead_steps=self.algo_lookahead_steps,
                n_actions=self.parms.n_actions,
                n_fantasy_at_design_pts=nf_design_pts,
                n_fantasy_at_action_pts=self.parms.n_samples,
                loss_function_class=self.parms.loss_function_class,
                loss_func_hypers=self.parms.loss_func_hypers,
                cost_function_class=self.parms.cost_function_class,
                cost_func_hypers=self.parms.cost_func_hypers,
            )

        elif self.parms.algo == "qKG":
            self.acqf = qKnowledgeGradient(
                model=WM, num_fantasies=self.parms.n_samples)

        elif self.parms.algo == "qEI":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            self.acqf = qExpectedImprovement(
                model=WM,
                best_f=buffer["y"].max(),
                sampler=sampler,
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
            self.acqf = qUpperConfidenceBound(
                model=WM, beta=0.1, sampler=sampler)

        elif self.parms.algo == "qMSL":
            num_fantasies = [self.parms.n_samples] * self.algo_lookahead_steps
            # num_fantasies = [self.parms.n_samples]
            # for s in range(1, self.algo_lookahead_steps):
            #     next_nf = num_fantasies[s-1] // 4
            #     if next_nf < 1:
            #         next_nf = 1
            #     num_fantasies.append(next_nf)

            self.acqf = qMultiStepLookahead(
                model=WM,
                batch_sizes=[1] * self.algo_lookahead_steps,
                num_fantasies=num_fantasies,
            )

        elif self.parms.aglo == "qNIPV":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
            self.acqf = qNegIntegratedPosteriorVariance(
                model=WM, mc_points=0, sampler=sampler  # TODO
            )

        elif self.parms.algo == "BudgetedBO":
            objective_new_x, cost_new_X = evaluate_obj_and_cost_at_X(
                X=buffer["x"],
                objective_function=self.parms.objective_function,
                cost_function=self.parms.cost_function,
                objective_cost_function=self.parms.objective_cost_function,
            )

            self.objective_X = torch.cat(
                [self.objective_X, objective_new_x], 0)
            self.cost_X = torch.cat([self.cost_X, cost_new_X], 0)

            suggested_budget, lower_bound = get_suggested_budget(
                strategy="fantasy_costs_from_aux_policy",
                refill_until_lower_bound_is_reached=self.parms.refill_until_lower_bound_is_reached,
                budget_left=self.acqf_params["budget_left"],
                model=WM,
                n_lookahead_steps=self.algo_lookahead_steps + 1,
                X=buffer["x"],
                objective_X=self.objective_X,
                cost_X=self.cost_X,
                init_budget=self.acqf_params["init_budget"],
                previous_budget=self.acqf_params["current_budget"],
                lower_bound=self.acqf_params["lower_bound"],
            )

            cumulative_cost = self.cost_X.sum().item()
            self.acqf_params["budget_left"] = (
                self.acqf_params["init_budget"] - cumulative_cost
            )
            self.acqf_params["current_budget"] = suggested_budget
            self.acqf_params["current_budget_plus_cumulative_cost"] = (
                suggested_budget + cumulative_cost
            )
            self.acqf_params["lower_bound"] = lower_bound

            num_fantasies = [self.parms.n_samples] * self.algo_lookahead_steps
            self.acqf = BudgetedMultiStepExpectedImprovement(
                model=WM,
                budget_plus_cumulative_cost=self.acqf_params[
                    "current_budget_plus_cumulative_cost"
                ],
                batch_size=1,
                lookahead_batch_sizes=[1] * self.algo_lookahead_steps,
                num_fantasies=num_fantasies,
            )

        else:
            raise ValueError(f"Unknown algo: {self.parms.algo}")

    def query(self, buffer, iteration: int, embedder=None, initial=False):
        r"""Compute the next design point.

        Args:
            prev_X (Tensor): Previous design points.
            prev_y (Tensor): Previous observations.
            prev_hid_state (Tensor): Previous hidden state.
            iteration: The current iteration.
        """
        assert self.acqf is not None, "Acquisition function is not initialized."

        prev_X = buffer["x"][iteration - 1: iteration].expand(
            self.parms.n_restarts, -1
        )

        if embedder is not None:
            # Discretize: Continuous -> Discrete
            prev_X = embedder.decode(prev_X)
            prev_X = torch.nn.functional.one_hot(
                prev_X, num_classes=self.parms.num_categories
            ).to(dtype=self.parms.torch_dtype)
            # >>> n_restarts x x_dim x n_categories

        prev_y = buffer["y"][iteration - 1: iteration].expand(
            self.parms.n_restarts, -1
        )
        prev_hid_state = buffer["h"][iteration - 1: iteration].expand(
            self.parms.n_restarts, -1
        )
        # Optimize the acquisition function
        if self.parms.algo == "HES":
            # Reset the actor parameters for diversity
            if iteration == self.parms.n_initial_points and initial:
                self.reset_parameters(
                    prev_X=prev_X,
                    prev_y=prev_y,
                    prev_hid_state=prev_hid_state,
                    embedder=embedder,
                )

                return_dict = self.acqf.forward(
                    prev_X=prev_X,
                    prev_y=prev_y,
                    prev_hid_state=prev_hid_state,
                    maps=self.maps,
                    embedder=embedder,
                )

                return return_dict["X"][0].detach(), None, return_dict["actions"].detach()

            optimizer = torch.optim.AdamW(
                self._parameters, lr=self.parms.acq_opt_lr)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=20, eta_min=1e-5
            )
            best_loss = torch.tensor([float("inf")], device=self.parms.device)
            best_next_X = None
            best_actions = None
            best_hidden_state = None
            best_map = None
            early_stop = 0
            losses = []
            costs = []

            self.acqf.dump_model()
            for ep in range(self.parms.acq_opt_iter):
                return_dict = self.acqf.forward(
                    prev_X=prev_X,
                    prev_y=prev_y,
                    prev_hid_state=prev_hid_state,
                    maps=self.maps,
                    embedder=embedder,
                )

                acqf_loss = return_dict["acqf_loss"].mean()
                acqf_cost = return_dict["acqf_cost"].mean()
                # >> n_restart

                losses.append(acqf_loss.item())
                costs.append(acqf_cost.item())
                loss = (return_dict["acqf_loss"] +
                        return_dict["acqf_cost"]).mean()

                if best_loss.mean() - loss > 0.1:
                    best_loss = return_dict["acqf_loss"] + \
                        return_dict["acqf_cost"]
                    best_next_X = return_dict["X"]
                    best_actions = return_dict["actions"]
                    best_hidden_state = return_dict["hidden_state"]
                    best_map = self.maps.state_dict()
                    early_stop = 0
                else:
                    if iteration > self.parms.n_initial_points or ep >= 200:
                        early_stop += 1
                grads = torch.autograd.grad(
                    loss, self._parameters, allow_unused=True)
                for param, grad in zip(self._parameters, grads):
                    param.grad = grad
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                print(f"Epoch {ep:05d}\tLoss {loss.item():.5f}",
                      end="\r", flush=True)
                if early_stop > 50:
                    break

            self.acqf.clean_dump_model()

            # Choose which restart produce the lowest loss
            idx = torch.argmin(best_loss)

            # Get next X as X_0 at idx
            next_X = best_next_X[0][idx].reshape(1, -1)

            # Get next hidden state of X_0 at idx
            hidden_state = best_hidden_state[0]
            # >>> n_restarts * hidden_dim

            if self.parms.amortized:
                hidden_state = hidden_state[idx: idx + 1]
                # >>> n_restarts * hidden_dim
                self.maps.load_state_dict(best_map)

            acqf_loss = best_loss[idx]
            # >>> n_actions * 1

            actions = best_actions[..., idx, :, :].detach()

            # Draw losses by acq_opt_iter
            draw_loss_and_cost(self.parms.save_dir, losses, costs, iteration)

        elif self.parms.algo == "BudgetedBO":
            bounds = self.parms.bounds.T

            next_X = optimize_acqf_and_get_suggested_point(
                acq_func=self.acqf,
                bounds=bounds,
                batch_size=1,
                algo_params=self.acqf_params,
            )
        else:
            # The total numbers of branches
            q = 1 + sum(
                [
                    self.parms.n_samples**s
                    for s in range(1, self.algo_lookahead_steps + 1)
                ]
            )

            if self.parms.env_name == "AntBO":
                choices = torch.tensor(
                    list(range(nm_AAs)), dtype=torch.long, device=self.parms.device
                )
                choices = choices.reshape(-1, 1).expand(-1, self.parms.x_dim)

                # Optimize acqf
                next_X, acqf_loss = optimize_acqf_discrete(
                    acq_function=self.acqf,
                    q=q,
                    choices=choices,
                )

            else:
                p_X = prev_X[-1]
                if embedder is not None:
                    p_X = embedder.encode(p_X)

                lb = p_X - self.parms.cost_func_hypers["radius"]
                ub = p_X + self.parms.cost_func_hypers["radius"]
                
                lb = torch.max(lb, self.parms.bounds[..., 0])
                ub = torch.min(ub, self.parms.bounds[..., 1])

                bounds = torch.stack([lb, ub], dim=0)

                # Optimize acqf
                next_X, acqf_loss = optimize_acqf(
                    acq_function=self.acqf,
                    bounds=bounds,
                    q=q,
                    num_restarts=self.parms.n_restarts,
                    raw_samples=self.parms.n_restarts,
                )

                if embedder is not None:
                    next_X = embedder.decode(next_X.reshape(1, -1))
                    next_X = torch.nn.functional.one_hot(
                        next_X, num_classes=self.parms.num_categories
                    ).to(dtype=self.parms.torch_dtype)
                    next_X = embedder.encode(next_X)

            hidden_state = prev_hid_state[0]
            actions = None

        next_X = next_X.detach()
        acqf_loss = acqf_loss.detach()
        hidden_state = hidden_state.detach()

        print("Acqf loss:", acqf_loss.item())

        return next_X, hidden_state, actions
