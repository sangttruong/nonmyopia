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
                    discrete=parms.discretized,
                    num_categories=self.parms.num_categories,
                )
                self.maps = self.maps.to(
                    dtype=self.parms.torch_dtype, device=self.parms.device
                )
                print(
                    "Number of AmortizedNet params:",
                    sum(p.numel() for p in self.maps.parameters() if p.requires_grad),
                )

                self._parameters = list(self.maps.parameters())

            else:
                self.maps = []

        # Initialize some actor attributes
        self.lookahead_steps = self.parms.lookahead_steps
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
            prev_X (Tensor): Previous design points
            prev_y (Tensor): Previous observations
        """
        print("Resetting actor parameters...")
        project2range = Project2Range(self.parms.bounds[0], self.parms.bounds[1])

        if self.parms.amortized:
            optimizer = torch.optim.AdamW(self._parameters, lr=self.parms.acq_opt_lr)

            for _ in tqdm(range(10)):
                optimizer.zero_grad()
                return_dict = self.acqf(
                    prev_X=prev_X,
                    prev_y=prev_y,
                    prev_hid_state=prev_hid_state,
                    maps=self.maps,
                    embedder=embedder,
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

                # loss.backward()
                grads = torch.autograd.grad(loss, self._parameters, allow_unused=True)
                for param, grad in zip(self._parameters, grads):
                    param.grad = grad
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
            if self.lookahead_steps == 1:
                nf_design_pts = [64]
            elif self.lookahead_steps == 2:
                nf_design_pts = [64, 64]
            elif self.lookahead_steps == 3:
                nf_design_pts = [64, 32, 8]
            elif self.lookahead_steps >= 4:
                nf_design_pts = [16, 8, 8, 8]
                nf_design_pts = nf_design_pts + [1] * (self.lookahead_steps - 4)

            # nf_design_pts = [self.parms.n_samples]
            # for s in range(1, self.lookahead_steps):
            #     next_nf = nf_design_pts[s-1] // 4
            #     if next_nf < 1:
            #         next_nf = 1
            #     nf_design_pts.append(next_nf)

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

        elif self.parms.algo == "qKG":
            self.acqf = qKnowledgeGradient(model=WM, num_fantasies=self.parms.n_samples)

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
            self.acqf = qUpperConfidenceBound(model=WM, beta=0.1, sampler=sampler)

        elif self.parms.algo == "qMSL":
            num_fantasies = [self.parms.n_samples] * self.lookahead_steps
            # num_fantasies = [self.parms.n_samples]
            # for s in range(1, self.lookahead_steps):
            #     next_nf = num_fantasies[s-1] // 4
            #     if next_nf < 1:
            #         next_nf = 1
            #     num_fantasies.append(next_nf)

            self.acqf = qMultiStepLookahead(
                model=WM,
                batch_sizes=[1] * self.lookahead_steps,
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

            self.objective_X = torch.cat([self.objective_X, objective_new_x], 0)
            self.cost_X = torch.cat([self.cost_X, cost_new_X], 0)

            suggested_budget, lower_bound = get_suggested_budget(
                strategy="fantasy_costs_from_aux_policy",
                refill_until_lower_bound_is_reached=self.parms.refill_until_lower_bound_is_reached,
                budget_left=self.acqf_params["budget_left"],
                model=WM,
                n_lookahead_steps=self.lookahead_steps + 1,
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

            num_fantasies = [self.parms.n_samples] * self.lookahead_steps
            self.acqf = BudgetedMultiStepExpectedImprovement(
                model=WM,
                budget_plus_cumulative_cost=self.acqf_params[
                    "current_budget_plus_cumulative_cost"
                ],
                batch_size=1,
                lookahead_batch_sizes=[1] * self.lookahead_steps,
                num_fantasies=num_fantasies,
            )

        else:
            raise ValueError(f"Unknown algo: {self.parms.algo}")

    def query(self, buffer, iteration: int, embedder=None):
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
        if embedder is not None:
            # Discretize: Continuous -> Discrete
            prev_X = embedder.decode(prev_X)
            prev_X = torch.nn.functional.one_hot(
                prev_X, num_classes=self.parms.num_categories
            ).to(dtype=self.parms.torch_dtype)
            # >>> n_restarts x x_dim x n_categories
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
                    prev_X=prev_X, prev_y=prev_y, prev_hid_state=prev_hid_state,
                    embedder=embedder,
                )

            optimizer = torch.optim.AdamW(self._parameters, lr=self.parms.acq_opt_lr)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=20, eta_min=1e-5
            )
            losses = []
            costs = []

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
                loss = (return_dict["acqf_loss"] + return_dict["acqf_cost"]).mean()
                
                # loss.backward()
                grads = torch.autograd.grad(loss, self._parameters, allow_unused=True)
                for param, grad in zip(self._parameters, grads):
                    param.grad = grad
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                print(f"Epoch {ep:05d}\tLoss {loss.item():.5f}", end="\r", flush=True)

            loss = return_dict["acqf_loss"] + return_dict["acqf_cost"]
            # Choose which restart produce the lowest loss
            idx = torch.argmin(loss)
            # Get next X as X_0 at idx
            next_X = return_dict["X"][0][idx].reshape(1, -1)
            if embedder is not None:
                next_X = embedder.decode(next_X)
                # next_X = torch.nn.functional.one_hot(
                #     next_X, num_classes=self.parms.num_categories).to(dtype=self.parms.torch_dtype)
                # next_X = embedder.encode(next_X)
                # >>> 1 * x_dim

            # Get next hidden state of X_0 at idx
            hidden_state = return_dict["hidden_state"][0]
            # >>> n_restarts * hidden_dim

            if self.parms.amortized:
                hidden_state = hidden_state[idx : idx + 1]
                # >>> n_restarts * hidden_dim

            acqf_loss = loss[idx]
            # >>> n_actions * 1

            actions = return_dict["actions"][..., idx, :, :].detach()

            # Draw losses by acq_opt_iter
            draw_loss_and_cost(self.parms.save_dir, losses, costs, iteration)

        elif self.parms.algo == "BudgetedBO":
            bounds = torch.tensor(
                [self.parms.bounds] * self.parms.x_dim,
                dtype=self.parms.torch_dtype,
                device=self.parms.device,
            ).T

            next_X = optimize_acqf_and_get_suggested_point(
                acq_func=self.acqf,
                bounds=bounds,
                batch_size=1,
                algo_params=self.acqf_params,
            )
        else:
            # The total numbers of branches
            q = 1 + sum(
                [self.parms.n_samples**s for s in range(1, self.lookahead_steps + 1)]
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
                # bounds = torch.tensor(
                #     [self.parms.bounds] * self.parms.x_dim,
                #     dtype=self.parms.torch_dtype,
                #     device=self.parms.device,
                # ).T
                p_X = prev_X[-1]
                if embedder is not None:
                    p_X = embedder.encode(p_X)

                lb = p_X - self.parms.cost_func_hypers["radius"]
                ub = p_X + self.parms.cost_func_hypers["radius"]

                lb[lb < self.parms.bounds[0]] = self.parms.bounds[0]
                ub[ub > self.parms.bounds[1]] = self.parms.bounds[1]

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
