#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement a amortized network."""

from typing import Tuple

import torch
import torch.nn as nn


class AmortizedNetwork(nn.Module):
    r"""Amortized network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_actions: int,
        output_bounds: Tuple[float, float],
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
        self.p = 0.2
        self.prepro = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(p=self.p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(p=self.p),
        )

        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.postpro_A = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(p=self.p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(p=self.p),
            nn.Linear(self.hidden_dim, self.output_dim * self.n_actions),
            Project2Range(self.output_bounds[0], self.output_bounds[1]),
        )

        self.postpro_X = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            Project2Range(self.output_bounds[0], self.output_bounds[1]),
        )

    def forward(self, x, prev_hid_state, return_actions):
        r"""Forward pass.

        Args:
            x: The input tensor.
            prev_hid_state: The previous hidden state.
            return_actions: Whether to return actions tensor or designs tensor.

        Returns:
            The output tensor and the hidden state.
        """
        postpro = self.postpro_A if return_actions else self.postpro_X
        preprocess_x = self.prepro(x)
        hidden_state = self.rnn(preprocess_x, prev_hid_state)
        preprocess_x = torch.cat([preprocess_x, hidden_state], dim=-1)
        return postpro(preprocess_x), hidden_state


class Project2Range(nn.Module):
    r"""Project the input to a range."""

    def __init__(self, min: int, max: int) -> None:
        r"""Initialize the module.

        Args:
            min: The minimum value of the range
            max: The maximum value of the range
        """
        super().__init__()
        self.min = min
        self.max = max
        self.range = self.max - self.min

    def forward(self, x):
        r"""Project the last dimension of the input to the range.

        Args:
            x: The (batch) input tensor.

        Returns:
            The projected tensor with the same dimention as the input.
        """
        return torch.sigmoid(x) * self.range + self.min
