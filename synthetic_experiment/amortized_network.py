#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement a amortized network."""

from typing import Tuple

import torch
import torch.nn as nn

from torch.distributions import OneHotCategoricalStraightThrough
from utils import generate_random_rotation_matrix


class AmortizedNetwork(nn.Module):
    r"""Amortized network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_actions: int,
        output_bounds: Tuple[float, float],
        discrete: bool = False,
        num_categories: int = 0,
    ) -> None:
        r"""Initialize the network.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            hidden_dim: The hidden dimension.
            n_actions: The number of actions.
            output_bounds: The bounds of the output.
        """
        super(AmortizedNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.output_bounds = output_bounds
        self.discrete = discrete
        self.num_categories = num_categories
        self.p = 0.0
        if self.discrete:
            self.embedding_x = nn.Linear(self.num_categories, self.hidden_dim)
            self.embedding_y = nn.Linear(1, self.hidden_dim)
            self.embedding = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        else:
            self.embedding = nn.Linear(self.input_dim, self.hidden_dim)

        self.prepro = nn.Sequential(
            nn.ELU(),
            nn.Dropout(p=self.p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(p=self.p),
        )

        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        output_action_dim = (
            self.output_dim * self.n_actions
            if not self.discrete
            else self.output_dim * self.n_actions * self.num_categories
        )
        self.postpro_A = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(p=self.p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(p=self.p),
            nn.Linear(self.hidden_dim, output_action_dim),
        )

        output_x_dim = (
            self.output_dim
            if not self.discrete
            else self.output_dim * self.num_categories
        )
        self.postpro_X = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(p=self.p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(p=self.p),
            nn.Linear(self.hidden_dim, output_x_dim),
        )

        self.project_output = Project2Range(
            self.output_bounds[..., 0], self.output_bounds[..., 1]
        )

    def forward(self, x, y, prev_hid_state, return_actions):
        r"""Forward pass.

        Args:
            x: The input tensor (batch, ).
            prev_hid_state: The previous hidden state.
            return_actions: Whether to return actions tensor or designs tensor.

        Returns:
            The output tensor and the hidden state.
        """
        postpro = self.postpro_A if return_actions else self.postpro_X

        if self.discrete:
            x = self.embedding_x(x)
            y = self.embedding_y(y)

            x = x.sum(dim=-2)
            # >>> batch x hidden_dim

        x = torch.cat([x, y], dim=-1)
        x = self.embedding(x)

        preprocess_x = self.prepro(x)
        hidden_state = self.rnn(preprocess_x, prev_hid_state)
        preprocess_x = torch.cat([preprocess_x, hidden_state], dim=-1)

        output = postpro(preprocess_x)
        if self.discrete:
            output = output.reshape(output.shape[0], -1, self.num_categories)
            output = output.softmax(dim=-1)
            return output, hidden_state

        output = self.project_output(output)

        angle_degrees = torch.randn(output.shape[0]) * 1e-1
        output = rotate_points(output, angle_degrees)

        return output, hidden_state


def rotate_points(inputs, angle_degrees):
    # Convert angle from degrees to radians
    thetas = torch.deg2rad(torch.tensor(angle_degrees))

    # Define the rotation matrix
    Q = [
        torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ]
        ).to(inputs)
        for theta in thetas
    ]
    Q = torch.stack(Q)

    D = inputs.shape[-1]
    dims = torch.stack([torch.randperm(D)[:2] for _ in thetas])
    M = torch.stack([torch.eye(D) for _ in thetas]).to(inputs)

    for i in range(thetas.shape[0]):
        M[i, dims[i, 0], dims[i, 0]] = Q[i, 0, 0]
        M[i, dims[i, 0], dims[i, 1]] = Q[i, 0, 1]
        M[i, dims[i, 1], dims[i, 0]] = Q[i, 1, 0]
        M[i, dims[i, 1], dims[i, 1]] = Q[i, 1, 1]

    rotated_points = torch.einsum("bd,bdd->bd", inputs, M.transpose(-1, -2))
    return rotated_points


class Project2Range(nn.Module):
    r"""Project the input to a range."""

    def __init__(self, _min, _max) -> None:
        r"""Initialize the module.

        Args:
            min: The minimum value of the range
            max: The maximum value of the range
        """
        super().__init__()
        self.min = _min
        self.max = _max
        self.range = self.max - self.min

    def forward(self, x):
        r"""Project the last dimension of the input to the range.

        Args:
            x: The (batch) input tensor.

        Returns:
            The projected tensor with the same dimention as the input.
        """
        # return torch.sigmoid(x) * self.range + self.min
        return torch.sigmoid(x)
