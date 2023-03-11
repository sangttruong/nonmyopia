from collections.abc import Iterable
import numpy as np
from scipy.spatial import distance
from scipy.stats import multivariate_normal


multihills_bounds = [0, 1]


def multihills(x_list, loc=None):
    """Multi-dimensional multihills function for multiple inputs."""
    if isinstance(x_list[0], Iterable):
        x_list_list = [list(x) for x in x_list]
        y_list = [multihills_single(x_list, loc) for x_list in x_list_list]
        return y_list
    else:
        x_list = list(x_list)
        y = multihills_single(x_list)
        return y

def multihills_single(x_list, loc=None):
    """Multi-dimensional multihills function for a single input."""
    if loc is None:
        d = len(x_list)
        loc = [[0.2] * d, [0.5] * d, [0.8] * d]

    dist_list = [-1 * distance.euclidean(lo, x_list)**0.35 for lo in loc]
    y = np.sum(dist_list)

    return y



class Multihills:

    def __init__(self, centers, widths, weights):
        """Initialize."""
        self.centers = centers
        self.widths = widths
        self.weights = weights
        self.init_components()

    def init_components(self):
        """Initialize components."""
        for center, width in zip(self.centers, self.widths):
            self.mvns = multivariate_normal(center, )

    def __call__(self, x_list):
        """Call on (potentially multiple) input."""
        if isinstance(x_list[0], Iterable):
            y_list = [self.call_single(xl) for xl in x_list]
            return y_list
        else:
            y = self.call_single(x_list)
            return y

    def call_single(self, x_list):
        """Call on single input."""
        vals = []
        for center, width, weight in zip(self.centers, self.widths, self.weights):
            pdf = multivariate_normal.pdf(x_list, mean=center, cov=width**2)
            vals.append(pdf * weight)

        sum_vals = np.sum(vals)
        return sum_vals
