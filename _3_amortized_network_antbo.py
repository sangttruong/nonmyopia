#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement a amortized network for AntBO."""

from typing import Tuple

import torch
import torch.nn as nn
from _9_semifuncs import nm_AAs


class AmortizedNetworkAntBO(nn.Module):
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
        super(AmortizedNetworkAntBO, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.output_bounds = output_bounds
        self.p = 0.2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=4, dropout=self.p, batch_first=True
        )

        self.prepro_X = nn.Sequential(
            nn.Linear(nm_AAs, self.hidden_dim),
            nn.TransformerEncoder(encoder_layer, num_layers=2),
        )

        self.prepo_y = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
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
            Project2Range(self.output_bounds[..., 0], self.output_bounds[..., 1]),
        )

        self.postpro_X = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            Project2Range(self.output_bounds[..., 0], self.output_bounds[..., 1]),
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
        x_onehot = nn.functional.one_hot(x[..., :-1].long(), num_classes=nm_AAs).float()
        preprocess_X = self.prepro_X(x_onehot)
        # >> batch x seq_length x hidden_dim

        preprocess_y = self.prepo_y(x[..., -1:])
        preprocess_y = preprocess_y[:, None, :].expand(-1, preprocess_X.shape[1], -1)
        # >> batch x seq_length x hidden_dim

        preprocess_Xy = torch.cat([preprocess_X, preprocess_y], dim=-1)
        desire_shape = preprocess_Xy.shape
        preprocess_Xy = preprocess_Xy.reshape(-1, self.hidden_dim)

        hidden_state = self.rnn(preprocess_Xy, prev_hid_state)
        ready_Xy = hidden_state.reshape(*desire_shape)

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
        self.min = torch.tensor(min)
        self.max = torch.tensor(max)
        self.range = self.max - self.min

    def forward(self, x):
        r"""Project the last dimension of the input to the range.

        Args:
            x: The (batch) input tensor.

        Returns:
            The projected tensor with the same dimention as the input.
        """
        return torch.sigmoid(x) * self.range + self.min
