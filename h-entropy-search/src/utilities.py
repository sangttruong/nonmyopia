"""
Various utilities.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def uniform_random_sample_domain(domain, n=1):
    """Draws a sample uniformly at random from domain (a list of tuple bounds)."""
    list_of_arr_per_dim = [np.random.uniform(dom[0], dom[1], n) for dom in domain]
    list_of_list_per_sample = [list(l) for l in np.array(list_of_arr_per_dim).T]
    return list_of_list_per_sample


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.main = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, X):
        return self.main(X)
