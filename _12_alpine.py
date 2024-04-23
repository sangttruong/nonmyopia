"""
Alpine N. 1 benchmark function.

Code is from: https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective
and: https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective/blob/d4dbc7eba1b746c7624c8bc41550054e828ca821/pybenchfunction/function.py#L224
"""

from collections.abc import Iterable

import numpy as np
from matplotlib import cm
import torch
from matplotlib import pyplot as plt
from torch import Tensor


cmap = [(0, "#2f9599"), (0.45, "#eeeeee"), (1, "#8800ff")]
cmap = cm.colors.LinearSegmentedColormap.from_list("Custom", cmap, N=256)


class AlpineN1:
    name = "Alpine N. 1"
    latex_formula = r"f(\mathbf x) = \sum_{i=1}^{d}|x_i sin(x_i)+0.1x_i|"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [0, 10], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=0"
    continuous = False
    convex = False
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def __init__(self, dim, x_scale=1.0, y_scale=1.0, noise_std=0.0, verbose=True):
        self.dim = dim
        self.input_domain = np.array([[0, 10] for _ in range(dim)])
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.noise_std = noise_std
        self.bounds = self.get_bounds(original=True)
        self.bounds = torch.tensor([self.bounds] * dim).T
        self.dtype = torch.float64
        if verbose:
            print(f"AlpineN1: x_scale={self.x_scale}, y_scale={self.y_scale}")

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(d)])
        return (X, self(X))

    def get_bounds(self, original=False):
        alpine_bounds = [0, 10]
        if original:
            return alpine_bounds
        else:
            return [self.x_scale * x for x in alpine_bounds]

    def transform_to_domain(self, x):
        # transform input to original scale
        new_x = x / self.x_scale
        return new_x

    def call_single(self, X):
        # transform input to original scale
        X = X / self.x_scale

        res = torch.sum(torch.abs(X * torch.sin(X) + 0.1 * X), dim=-1)
        return res

    def call_tensor(self, x_list):
        assert x_list.shape[-1] == self.dim
        y_tensor = self.call_single(x_list)
        return y_tensor

    def __call__(self, x):
        y = self.call_tensor(x)

        # transform y from original to new scale
        y = y * self.y_scale

        return y.to(x.device, self.dtype)

    def to(self, dtype, device):
        self.dtype = dtype
        self.device = device
        return self


def plot_alpine_2d(n_space=200, cmap=cmap, XYZ=None, ax=None, threshold=None):
    """Plot AlpineN1 function in 2d."""
    alpine = AlpineN1(dim=2, verbose=False)
    X_domain, Y_domain = alpine.input_domain
    if XYZ is None:
        X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
        X, Y = np.meshgrid(X, Y)
        XY = np.array([X, Y])
        Z = np.apply_along_axis(alpine, 0, XY)
    else:
        X, Y, Z = XYZ

    # add contours and contours lines
    # ax.contour(X, Y, Z, levels=30, linewidths=0.5, colors='#999')
    ax.contourf(X, Y, Z, levels=30, cmap=cmap, alpha=0.7)
    ax.set_aspect(aspect="equal")

    if threshold:
        if type(threshold) is list:
            assert len(threshold) == 2
            ax.contour(
                X,
                Y,
                Z,
                levels=threshold,
                colors=["blue", "red"],
                linewidths=3,
                linestyles="dashed",
            )
        else:
            ax.contour(
                X,
                Y,
                Z,
                levels=[threshold],
                colors="red",
                linewidths=3,
                linestyles="dashed",
            )


# Select level set threshold
# fig, ax = plt.subplots(figsize=(6, 6))
# plot_alpine_2d(ax=ax, threshold=0.45 / 0.05)
# plt.show()
