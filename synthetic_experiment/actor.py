#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement an actor."""
import gc
import os
import pickle

import torch
from acqfs import qBOAcqf, qMultiStepHEntropySearch
from amortized_network import AmortizedNetwork

from botorch.sampling.normal import SobolQMCNormalSampler
from utils import (
    draw_loss_and_cost,
    generate_random_points_batch,
    generate_random_rotation_matrix,
    rotate_points,
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
                ).to(dtype=self.parms.torch_dtype, device=self.parms.device)
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
        # Inititalize required variables
        prev_X = buffer["x"][-1:].expand(self.parms.n_restarts, -1)
        prev_y = buffer["y"][-1:].expand(self.parms.n_restarts, -1)

        if self.parms.algo_ts:
            nf_design_pts = [1] * self.algo_lookahead_steps
        elif "MSL" not in self.parms.algo:
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
        else:
            nf_design_pts = [32, 2, 1, 1][: self.algo_lookahead_steps]

        if self.parms.amortized:
            prev_hid_state = buffer["h"][-1:].clone().expand(self.parms.n_restarts, -1)
            optimizer = torch.optim.AdamW(
                self._parameters, lr=0.01
            )  # self.parms.acq_opt_lr)
            n_samples = self.parms.n_restarts  # 1000
            d = self.parms.x_dim

            X = []
            for s in range(self.algo_lookahead_steps):
                if s == 0:
                    prev_points = prev_X[0:1]
                    n_points = n_samples
                else:
                    prev_points = X[-1]
                    n_points = nf_design_pts[s - 1]

                x = generate_random_points_batch(
                    prev_points, self.parms.cost_func_hypers["radius"], n_points
                )
                x = torch.clamp(
                    x,
                    max=0.99,
                    min=0.01,
                )

                X.append(
                    x.to(
                        device=self.parms.device,
                        dtype=self.parms.torch_dtype,
                    ).reshape(-1, self.parms.x_dim)
                )

            if len(X) == 0:
                prev_points = prev_X[0:1]
                n_points = n_samples * self.parms.n_actions
            else:
                prev_points = X[-1]
                n_points = nf_design_pts[-1] * self.parms.n_actions

            A = (
                generate_random_points_batch(
                    prev_points, self.parms.cost_func_hypers["radius"], n_points
                )
                .to(
                    device=self.parms.device,
                    dtype=self.parms.torch_dtype,
                )
                .reshape(-1, self.parms.x_dim)
            )

            X.append(
                torch.clamp(
                    A,
                    max=0.99,
                    min=0.01,
                ).detach()
            )
            X = torch.stack(X, dim=0)

            self.hidden_noise = torch.randn(n_samples, self.parms.hidden_dim).to(X[-1])
            prev_hid_state = prev_hid_state + self.hidden_noise

            early_stop_count = 0
            best_loss = float("inf")
            for ep in range(300):
                outputs = []
                local_prev_hid_state = prev_hid_state
                prev = prev_X[0].expand(n_samples, self.parms.x_dim)
                y = prev_y[0].expand(n_samples, self.parms.y_dim)

                for j in range(self.algo_lookahead_steps):
                    output, hidden_state = self.maps(
                        prev,
                        y,
                        local_prev_hid_state,
                        return_actions=False,
                        enable_noise=False,
                    )
                    outputs.append(output)
                    prev = output
                    y = (
                        self.acqf.model(output.unsqueeze(-2))
                        .sample(sample_shape=torch.Size([nf_design_pts[j]]))
                        .reshape(-1, 1)
                    )
                    local_prev_hid_state = hidden_state

                output, hidden_state = self.maps(
                    prev,
                    y,
                    local_prev_hid_state,
                    return_actions=True,
                    enable_noise=False,
                )
                outputs.append(output)

                outputs = torch.stack(outputs, dim=0)
                loss = torch.mean(abs(X - outputs))

                if ep % 10 == 0:
                    print("Loss:", loss.item())

                if loss < best_loss:
                    best_loss = loss
                    early_stop_count = 0
                else:
                    early_stop_count += 1

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if early_stop_count >= 50:
                    break

        else:
            if bo_iter == 0 or not self.parms.algo_ts:
                self.maps = []

                local_maps = []
                for s in range(self.algo_lookahead_steps):
                    if s == 0:
                        prev_points = prev_X[0]
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
                    prev_points = prev_X[0]
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
                if embedder is not None:
                    self.maps = [
                        embedder.encode(x.requires_grad_(False)) for x in self.maps
                    ]
                    local_maps = []
                    for maps in self.maps:
                        local_maps.append(
                            torch.nn.functional.one_hot(
                                embedder.decode(maps),
                                num_classes=self.parms.num_categories,
                            ).to(self.parms.torch_dtype)
                        )
                    self.maps = [embedder.encode(x) for x in local_maps]
                else:
                    self.maps = [
                        torch.sigmoid(x.requires_grad_(False)) for x in self.maps
                    ]

                # 1. Pick the best trajectory
                prev_chosen_idx = prev_chosen_idx.long()
                prev_points = prev_X[0]
                # >>> n_restarts x x_dim

                random_R_matrices = [torch.eye(self.parms.x_dim).to(prev_X)]
                random_R_matrices.extend(
                    [
                        generate_random_rotation_matrix(self.parms.x_dim).to(prev_X)
                        for _ in range(self.parms.n_restarts - 1)
                    ]
                )
                random_R_matrices = torch.stack(random_R_matrices, dim=0)
                # >>> n_restarts x x_dim x x_dim

                ##### Work for TS only #####
                self.maps[0][:] = self.maps[1][prev_chosen_idx]

                for lah in range(2, self.algo_lookahead_steps + 1):
                    lah_points = self.maps[lah].reshape(
                        *nf_design_pts[:lah], self.parms.n_restarts, self.parms.x_dim
                    )
                    best_traj_lah = lah_points[..., prev_chosen_idx, :]
                    # >>> ... x x_dim

                    list_rotated = rotate_points(
                        best_traj_lah.reshape(-1, self.parms.x_dim),
                        random_R_matrices,
                        self.maps[0][prev_chosen_idx],
                    ).transpose(0, 1)

                    list_rotated = torch.clamp(
                        list_rotated,
                        max=0.99,
                        min=0.01,
                    )

                    self.maps[lah - 1] = list_rotated.reshape(-1, self.parms.x_dim)

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
                if embedder is not None:
                    self.maps = [
                        torch.nn.functional.one_hot(
                            embedder.decode(x), num_classes=self.parms.num_categories
                        )
                        .to(self.parms.torch_dtype)
                        .requires_grad_(True)
                        for x in self.maps
                    ]
                else:
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
        elif "MSL" not in self.parms.algo:
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
        else:
            nf_design_pts = [32, 2, 1, 1][: self.algo_lookahead_steps]

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

    def query(
        self,
        buffer,
        iteration: int,
        embedder=None,
        f_posterior=None,
        save_trajectory=True,
    ):
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
        if self.parms.amortized:
            # if iteration - self.parms.n_initial_points > 0:
            #     self.hidden_noise = torch.randn(self.parms.n_restarts, self.parms.hidden_dim).to(prev_X) * 1e-2
            #     self.hidden_noise[0] = 0.0
            # if iteration - self.parms.n_initial_points == 0:
            if True:
                prev_hid_state = prev_hid_state + self.hidden_noise
        prev_cost = (
            buffer["cost"][self.parms.n_initial_points : iteration].sum()
            if iteration > self.parms.n_initial_points
            else 0.0
        )

        # Optimize the acquisition function
        optimizer = torch.optim.AdamW(self._parameters, lr=self.parms.acq_opt_lr)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.parms.acq_opt_iter
        # )
        best_loss = torch.tensor([float("inf")], device=self.parms.device)
        best_cost = torch.tensor([float("inf")], device=self.parms.device)
        best_next_X = None
        best_actions = None
        best_hidden_state = None
        best_map = None
        best_KL = 0
        early_stop = 0
        losses = []
        costs = []

        if self.parms.algo.startswith("HES"):
            self.acqf.dump_model(f=f_posterior)
            if self.parms.algo_ts and f_posterior is None:
                # Save self.acqf.f using pickle
                pickle.dump(
                    self.acqf.f,
                    open(
                        os.path.join(
                            self.parms.save_dir, f"posterior_sample_{iteration}.pkl"
                        ),
                        "wb",
                    ),
                )

        ########## TESTING ###########
        saved_trajectory = []
        saved_loss = []
        saved_cost = []
        ##############################

        if self.parms.amortized:
            original_sd = self.maps.state_dict()
            original_maps = AmortizedNetwork(
                input_dim=self.parms.x_dim + self.parms.y_dim,
                output_dim=self.parms.x_dim,
                hidden_dim=self.parms.hidden_dim,
                n_actions=self.parms.n_actions,
                output_bounds=self.parms.bounds,
                discrete=self.parms.env_discretized,
                num_categories=self.parms.num_categories,
            ).to(dtype=self.parms.torch_dtype, device=self.parms.device)
            original_maps.load_state_dict(original_sd)
            cosine_loss_fn = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        ########## TESTING ###########
        # saved_trajectory = []
        # saved_loss = []
        # saved_cost = []
        ##############################

        if self.parms.amortized:
            original_sd = self.maps.state_dict()
            original_maps = AmortizedNetwork(
                input_dim=self.parms.x_dim + self.parms.y_dim,
                output_dim=self.parms.x_dim,
                hidden_dim=self.parms.hidden_dim,
                n_actions=self.parms.n_actions,
                output_bounds=self.parms.bounds,
                discrete=self.parms.env_discretized,
                num_categories=self.parms.num_categories,
            ).to(dtype=self.parms.torch_dtype, device=self.parms.device)
            original_maps.load_state_dict(original_sd)
            cosine_loss_fn = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        for ep in range(self.parms.acq_opt_iter):
            if not self.parms.amortized and not self.parms.env_discretized:
                local_maps = [torch.sigmoid(x) for x in self.maps]
                if save_trajectory:
                    saved_trajectory.append(
                        [x.cpu().detach().tolist() for x in local_maps]
                    )
            else:
                local_maps = self.maps

            return_dict = self.acqf.forward(
                prev_X=prev_X,
                prev_y=prev_y,
                prev_hid_state=prev_hid_state,
                maps=local_maps,
                embedder=embedder,
                prev_cost=prev_cost,
                enable_noise=False,  # (iteration > self.parms.n_initial_points),
            )

            acqf_loss = return_dict["acqf_loss"]
            acqf_cost = return_dict["acqf_cost"]
            saved_loss.append(return_dict["acqf_loss"].tolist())
            saved_cost.append(return_dict["acqf_cost"].tolist())
            # >> n_restart

            if self.parms.amortized:
                with torch.no_grad():
                    original_return_dict = self.acqf.forward(
                        prev_X=prev_X,
                        prev_y=prev_y,
                        prev_hid_state=prev_hid_state,
                        maps=original_maps,
                        embedder=embedder,
                        prev_cost=prev_cost,
                    )
                KL = 0
                for lhi in range(self.algo_lookahead_steps):
                    KL = (
                        KL
                        + 1
                        - cosine_loss_fn(
                            original_return_dict["X"][lhi], return_dict["X"][lhi]
                        ).flatten()
                    )
                KL = (
                    KL
                    + 1
                    - cosine_loss_fn(
                        original_return_dict["actions"], return_dict["actions"]
                    ).flatten()
                )
            else:
                KL = 0

            if save_trajectory and (self.parms.amortized or self.parms.env_discretized):
                saved_trajectory.append(
                    [x.cpu().detach().tolist() for x in return_dict["X"]]
                )

            losses.append(acqf_loss.mean().item())
            costs.append(acqf_cost.mean().item())
            loss = acqf_loss + abs(self.parms.budget - acqf_cost) + KL

            chosen_idx = torch.argmin(loss)
            if ep == 0 or loss[chosen_idx] < (best_loss + best_cost + best_KL).min():
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
                    best_KL = KL.detach()
                    best_map = self.maps.state_dict()
                early_stop = 0
            else:
                if iteration > self.parms.n_initial_points or ep >= 200:
                    early_stop += 1

            loss = loss.mean()
            grads = torch.autograd.grad(loss, self._parameters, allow_unused=True)
            for param, grad in zip(self._parameters, grads):
                param.grad = grad

            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()

            if ep % 50 == 0:
                print(
                    f"Epoch {ep:05d}\tLoss {acqf_loss[chosen_idx].item():.5f}\tCost: {acqf_cost[chosen_idx].item():.5f}"
                )

            if early_stop > 50:
                break

        if self.parms.algo.startswith("HES"):
            self.acqf.clean_dump_model()

        # Choose which restart produce the lowest loss
        idx = torch.argmin(best_loss + best_cost + best_KL)

        ########## TESTING ###########
        if save_trajectory:
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
        if embedder is not None:
            # Discretize: Continuous -> Discrete
            next_X = embedder.decode(next_X)
            next_X = torch.nn.functional.one_hot(
                next_X, num_classes=self.parms.num_categories
            ).to(dtype=self.parms.torch_dtype)
            # >>> n_restarts x x_dim x n_categories

            # Cat ==> Con
            next_X = embedder.encode(next_X)

            # Discretize: Continuous -> Discrete
            actions = embedder.decode(actions)
            actions = torch.nn.functional.one_hot(
                actions, num_classes=self.parms.num_categories
            ).to(dtype=self.parms.torch_dtype)
            # >>> n_restarts x x_dim x n_categories

            # Cat ==> Con
            actions = embedder.encode(actions)

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
            "all_losses": best_loss,
            "all_costs": best_cost,
            "loss": acqf_loss,
            "cost": acqf_cost,
            "next_X": next_X,
            "actions": actions,
            "hidden_state": hidden_state,
            "chosen_idx": idx,
        }
