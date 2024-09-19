#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement an actor."""
import gc
import math
import pickle

import torch
from acqfs import qBOAcqf, qMultiStepHEntropySearch
from amortized_network import AmortizedNetwork, Project2Range

from botorch.sampling.normal import SobolQMCNormalSampler
from tqdm import tqdm
from utils import (
    draw_loss_and_cost,
    generate_random_points_batch,
    rotate_around_point_highperf,
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
            self.acqf_class = qMultiStepHEntropySearch

            if self.parms.amortized:
                self.maps = AmortizedNetwork(
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
                    sum(p.numel() for p in self.maps.parameters() if p.requires_grad),
                )

                self._parameters = list(self.maps.parameters())

            else:
                self.maps = []
        else:
            self.acqf_class = qBOAcqf
            self.maps = []

        # Initialize some actor attributes
        self.algo_lookahead_steps = self.parms.algo_lookahead_steps
        self.acqf = None

    def reset_parameters(
        self,
        buffer,
        bo_iter=0,
        embedder=None,
        prev_chosen_idx=0,
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
        # Inititalize required variables
        prev_X = buffer["x"][-1:].expand(self.parms.n_restarts, -1)
        if embedder is not None:
            # Discretize: Continuous -> Discrete
            prev_X = embedder.decode(prev_X)
            prev_X = torch.nn.functional.one_hot(
                prev_X, num_classes=self.parms.num_categories
            ).to(dtype=self.parms.torch_dtype)
            # >>> n_restarts x x_dim x n_categories

            # Cat ==> Con
            encoded_prev_X = embedder.encode(prev_X)
        else:
            encoded_prev_X = prev_X

        prev_y = buffer["y"][-1:].expand(self.parms.n_restarts, -1)
        prev_hid_state = buffer["h"][-1:].expand(self.parms.n_restarts, -1)

        if self.parms.amortized:
            optimizer = torch.optim.AdamW(self._parameters, lr=self.parms.acq_opt_lr)

            # X_randomizeds = []
            # for s in range(self.algo_lookahead_steps):
            #     if s == 0:
            #         prev_points = encoded_prev_X[0]
            #         n_points = 1000
            #     else:
            #         prev_points = X_randomizeds[-1]
            #         n_points = 1

            #     X_randomized = generate_random_points_batch(
            #         prev_points, self.parms.cost_func_hypers["radius"], n_points
            #     )
            #     X_randomizeds.append(X_randomized.reshape(-1, self.parms.x_dim).to(
            #         device=self.parms.device,
            #         dtype=self.parms.torch_dtype,
            #     ))

            # A_randomized = generate_random_points_batch(
            #     X_randomizeds[-1],
            #     self.parms.cost_func_hypers["radius"],
            #     1,
            # ).reshape(-1, self.parms.x_dim).to(
            #     device=self.parms.device,
            #     dtype=self.parms.torch_dtype,
            # )

            A_randomized = torch.rand(
                (1000, self.parms.x_dim),
                device=self.parms.device,
                dtype=self.parms.torch_dtype,
            )
            ub = torch.clamp(
                encoded_prev_X[0]
                + self.parms.cost_func_hypers["radius"]
                * (self.algo_lookahead_steps + 1),
                max=1,
            )
            lb = torch.clamp(
                encoded_prev_X[0]
                - self.parms.cost_func_hypers["radius"]
                * (self.algo_lookahead_steps + 1),
                min=0,
            )
            A_randomized = (A_randomized * (ub - lb) + lb).detach()

            X_randomizeds = []
            for i in range(self.algo_lookahead_steps):
                X_randomized = torch.rand(
                    (1000, self.parms.x_dim),
                    device=self.parms.device,
                    dtype=self.parms.torch_dtype,
                )
                ub = torch.clamp(
                    encoded_prev_X[0] + self.parms.cost_func_hypers["radius"] * (i + 1),
                    max=1,
                )
                lb = torch.clamp(
                    encoded_prev_X[0] - self.parms.cost_func_hypers["radius"] * (i + 1),
                    min=0,
                )

                X_randomized = (X_randomized * (ub - lb) + lb).detach()
                X_randomizeds.append(X_randomized)

            std = self.parms.cost_func_hypers["radius"]
            self.acqf.dump_model()

            for _ in tqdm(range(100)):
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

                grads = torch.autograd.grad(loss, self._parameters, allow_unused=True)
                for param, grad in zip(self._parameters, grads):
                    param.grad = grad
                optimizer.step()

            self.acqf.clean_dump_model()

        else:
            if self.parms.algo_ts:
                nf_design_pts = [1] * self.algo_lookahead_steps
            else:
                if self.algo_lookahead_steps == 0:
                    nf_design_pts = []
                elif self.algo_lookahead_steps == 1:
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

            if bo_iter == 0 or not self.parms.algo_ts:
                self.maps = []

                local_maps = []
                for s in range(self.algo_lookahead_steps):
                    if s == 0:
                        prev_points = encoded_prev_X[0]
                        n_points = self.parms.n_restarts
                    else:
                        prev_points = local_maps[-1]
                        n_points = nf_design_pts[s - 1]

                    x = generate_random_points_batch(
                        prev_points, self.parms.cost_func_hypers["radius"], n_points
                    )
                    local_maps.append(x)

                for s in range(self.algo_lookahead_steps):
                    x = (
                        local_maps[s]
                        .reshape(-1, self.parms.x_dim)
                        .to(
                            device=self.parms.device,
                            dtype=self.parms.torch_dtype,
                        )
                    )
                    if embedder is not None:
                        x = embedder.decode(x)
                        x = torch.nn.functional.one_hot(
                            x, num_classes=self.parms.num_categories
                        ).to(self.parms.torch_dtype)
                    else:
                        x = torch.clamp(
                            x,
                            max=0.99,
                            min=0.01,
                        )
                        x = -torch.log(1 / x - 1)
                    self.maps.append(x.requires_grad_(True))

                if len(local_maps) == 0:
                    prev_points = encoded_prev_X[0]
                    n_points = self.parms.n_restarts * self.parms.n_actions
                else:
                    prev_points = local_maps[-1]
                    n_points = nf_design_pts[-1] * self.parms.n_actions

                a = (
                    generate_random_points_batch(
                        prev_points, self.parms.cost_func_hypers["radius"], n_points
                    )
                    .reshape(-1, self.parms.x_dim)
                    .to(
                        device=self.parms.device,
                        dtype=self.parms.torch_dtype,
                    )
                )

                if embedder is not None:
                    a = embedder.decode(a)
                    a = torch.nn.functional.one_hot(
                        a, num_classes=self.parms.num_categories
                    ).to(self.parms.torch_dtype)
                else:
                    a = torch.clamp(
                        a,
                        max=0.99,
                        min=0.01,
                    )
                    a = -torch.log(1 / a - 1)
                self.maps.append(a.requires_grad_(True))
            else:
                # Rotate previous best trajectory
                self.maps = [torch.sigmoid(x.requires_grad_(False)) for x in self.maps]
                # 1. Pick the best trajectory
                prev_chosen_idx = prev_chosen_idx.long()
                prev_points = encoded_prev_X[0]
                # >>> n_restarts x x_dim

                random_angles = torch.zeros(self.parms.n_restarts).to(encoded_prev_X)
                random_angles[1:] = torch.deg2rad(
                    torch.randint(low=0, high=360, size=(self.parms.n_restarts - 1,))
                )
                # >>> n_restarts

                ##### Work for TS only #####
                self.maps[0][:] = self.maps[1][prev_chosen_idx]

                for lah in range(2, self.algo_lookahead_steps + 1):
                    lah_points = self.maps[lah].reshape(
                        *nf_design_pts[:lah], self.parms.n_restarts, self.parms.x_dim
                    )
                    best_traj_lah = lah_points[..., prev_chosen_idx, :]
                    # >>> ... x 1 x x_dim

                    list_rotated = []
                    for ridx, angle in enumerate(random_angles):
                        new_rotated_points = rotate_around_point_highperf(
                            best_traj_lah.reshape(-1, self.parms.x_dim),
                            angle,
                            self.maps[0][prev_chosen_idx],
                        )

                        new_rotated_points = new_rotated_points.reshape(
                            *nf_design_pts[:lah], 1, self.parms.x_dim
                        )
                        new_rotated_points = torch.clamp(
                            new_rotated_points,
                            max=0.99,
                            min=0.01,
                        )
                        list_rotated.append(new_rotated_points)

                    self.maps[lah - 1] = torch.stack(list_rotated, dim=-2).reshape(
                        -1, self.parms.x_dim
                    )

                a = generate_random_points_batch(
                    self.maps[self.algo_lookahead_steps - 1],
                    self.parms.cost_func_hypers["radius"],
                    nf_design_pts[-1] * self.parms.n_actions,
                ).to(
                    device=self.parms.device,
                    dtype=self.parms.torch_dtype,
                )
                a = torch.clamp(
                    a,
                    max=0.99,
                    min=0.01,
                )
                self.maps[self.algo_lookahead_steps] = a.reshape(-1, self.parms.x_dim)
                self.maps = [
                    (-torch.log(1 / x - 1)).requires_grad_(True) for x in self.maps
                ]
            self._parameters = self.maps

    def construct_acqf(self, surr_model, buffer, **kwargs):
        """Contruct aquisition function.

        Args:
            surr_model: Surrogate model.
            buffer: A ReplayBuffer object containing the data.

        Raises:
            ValueError: If defined algo is not implemented.

        Returns:
            AcquisitionFunction: An aquisition function instance
        """
        del self.acqf
        gc.collect()
        torch.cuda.empty_cache()

        if self.parms.algo_ts:
            nf_design_pts = [1] * self.algo_lookahead_steps
        else:
            if self.algo_lookahead_steps == 0:
                nf_design_pts = []
            elif self.algo_lookahead_steps == 1:
                nf_design_pts = [64]
            elif self.algo_lookahead_steps == 2:
                nf_design_pts = [64, 8]  # [64, 64]
            elif self.algo_lookahead_steps == 3:
                nf_design_pts = [64, 4, 2]  # [64, 32, 8]
            elif self.algo_lookahead_steps >= 4:
                nf_design_pts = [64, 4, 2, 1]  # [16, 8, 8, 8]
                nf_design_pts = nf_design_pts + [1] * (self.algo_lookahead_steps - 4)

        if self.parms.algo != "HES":
            sampler = SobolQMCNormalSampler(
                sample_shape=self.parms.n_samples, seed=0, resample=False
            )
        else:
            sampler = None

        self.acqf = self.acqf_class(
            name=self.parms.algo,
            model=surr_model,
            lookahead_steps=self.algo_lookahead_steps,
            n_actions=self.parms.n_actions,
            n_fantasy_at_design_pts=nf_design_pts,
            n_fantasy_at_action_pts=self.parms.n_samples,
            loss_function_class=self.parms.loss_function_class,
            loss_func_hypers=self.parms.loss_func_hypers,
            cost_function_class=self.parms.cost_function_class,
            cost_func_hypers=self.parms.cost_func_hypers,
            enable_ts=self.parms.algo_ts,
            sampler=sampler,
            best_f=buffer["y"].max(),
        )

    def query(self, buffer, iteration: int, embedder=None, initial=False):
        r"""Compute the next design point.

        Args:
            prev_X (Tensor): Previous design points.
            prev_y (Tensor): Previous observations.
            prev_hid_state (Tensor): Previous hidden state.
            iteration: The current iteration.
        """
        assert self.acqf is not None, "Acquisition function is not initialized."

        # Inititalize required variables
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
        prev_cost = (
            buffer["cost"][self.parms.n_initial_points : iteration].sum()
            if iteration > self.parms.n_initial_points
            else 0.0
        )

        # Optimize the acquisition function
        # optimizer = torch.optim.LBFGS(self._parameters, lr=self.parms.acq_opt_lr)
        optimizer = torch.optim.AdamW(self._parameters, lr=self.parms.acq_opt_lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.parms.acq_opt_iter
        )
        best_loss = torch.tensor([float("inf")], device=self.parms.device)
        best_cost = torch.tensor([float("inf")], device=self.parms.device)
        best_next_X = None
        best_actions = None
        best_hidden_state = None
        best_map = None
        early_stop = 0
        losses = []
        costs = []

        if self.parms.algo.startswith("HES"):
            self.acqf.dump_model()

        ########## TESTING ###########
        saved_trajectory = []
        saved_loss = []
        saved_cost = []
        ##############################

        for ep in range(self.parms.acq_opt_iter):
            if not self.parms.amortized and not self.parms.env_discretized:
                local_maps = [torch.sigmoid(x) for x in self.maps]
                saved_trajectory.append([x.cpu().detach().tolist() for x in local_maps])
            else:
                local_maps = self.maps

            return_dict = self.acqf.forward(
                prev_X=prev_X,
                prev_y=prev_y,
                prev_hid_state=prev_hid_state,
                maps=local_maps,
                embedder=embedder,
                prev_cost=prev_cost,
            )

            acqf_loss = return_dict["acqf_loss"]
            acqf_cost = return_dict["acqf_cost"]
            saved_loss.append(return_dict["acqf_loss"].tolist())
            saved_cost.append(return_dict["acqf_cost"].tolist())
            # >> n_restart

            losses.append(acqf_loss.mean().item())
            costs.append(acqf_cost.mean().item())
            loss = (acqf_loss + acqf_cost).mean()

            if loss < (best_loss + best_cost).mean():
                best_loss = return_dict["acqf_loss"].detach()
                best_cost = return_dict["acqf_cost"].detach()
                best_next_X = [x.detach() for x in return_dict["X"]]
                best_actions = return_dict["actions"].detach()
                best_hidden_state = (
                    [x.detach() for x in return_dict["hidden_state"]]
                    if return_dict["hidden_state"]
                    else None
                )
                if self.parms.amortized:
                    best_map = self.maps.state_dict()
                early_stop = 0
            else:
                if iteration > self.parms.n_initial_points or ep >= 200:
                    early_stop += 1

            grads = torch.autograd.grad(loss, self._parameters, allow_unused=True)
            for param, grad in zip(self._parameters, grads):
                param.grad = grad

            # optimizer.step(lambda: loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if ep % 50 == 0:
                idx = torch.argmin(acqf_loss + acqf_cost)
                print(
                    f"Epoch {ep:05d}\tLoss {acqf_loss[idx].item():.5f}\tCost: {acqf_cost[idx].item():.5f}"
                )

            if early_stop > 50:
                break

        if self.parms.algo.startswith("HES"):
            self.acqf.clean_dump_model()

        # Choose which restart produce the lowest loss
        idx = torch.argmin(best_loss + best_cost)

        ########## TESTING ###########
        pickle.dump(
            (saved_trajectory, idx),
            open(self.parms.save_dir + f"/trajectory_{iteration}.pkl", "wb"),
        )
        pickle.dump(
            (saved_loss, saved_cost),
            open(self.parms.save_dir + f"/lossncost_{iteration}.pkl", "wb"),
        )
        ##############################

        # Best acqf loss
        acqf_loss = best_loss[idx]
        # >>> n_actions * 1

        # Get next X as X_0 at idx
        next_X = best_next_X[0][idx].reshape(1, -1)

        # Get best actions
        actions = best_actions[..., idx, :, :]

        # Get next hidden state of X_0 at idx
        if self.parms.amortized:
            hidden_state = best_hidden_state[0][idx : idx + 1]
            # >>> n_restarts * hidden_dim
            self.maps.load_state_dict(best_map)
        else:
            hidden_state = None

        # Compute acqf loss
        acqf_cost = self.acqf.cost_function(
            prev_X=buffer["x"][iteration - 1 : iteration],
            current_X=next_X,
            previous_cost=prev_cost,
        ).detach()

        # Draw losses by acq_opt_iter
        if self.parms.plot:
            draw_loss_and_cost(self.parms.save_dir, losses, costs, iteration)

        return {
            "cost": acqf_cost,
            "next_X": next_X,
            "actions": actions,
            "hidden_state": hidden_state,
            "chosen_idx": idx,
        }
