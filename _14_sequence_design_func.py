import torch
import math

import numpy as np

from torch import Tensor


class SequenceDesignFunction:
    def __init__(self, dim, x_scale=1.0, y_scale=1.0):
        self.dim = dim
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.bounds = self.get_bounds(original=True)
        self.bounds = torch.tensor([self.bounds] * dim).T

    def get_bounds(self, original=False):
        bounds = [0, self.dim]
        if original:
            return bounds
        else:
            return [self.x_scale * x for x in bounds]

    def call_single(self, x):
        unique, counts = np.unique(x, return_counts=True)
        counts = dict(zip(unique, counts))
        number_b = counts.get(1, 0)
        number_c = counts.get(2, 0)
        threshold = int(math.ceil(len(x) / 2))
        scores = torch.zeros(1)
        if number_c == self.dim:
            scores[0] = len(x) + 1
        else:
            if number_c >= threshold:
                score_c = ((len(x) + 1 + threshold)) / ((len(x) - threshold))
                scores[0] = number_b - threshold + (number_c % threshold) * (score_c)
            else:
                scores[0] = number_b - number_c
        return scores

    def call_tensor(self, x_list):
        x_list = [xi.cpu().detach().numpy().tolist() for xi in x_list]
        x_list = np.array(x_list).reshape(-1, self.dim)
        y_list = [self.call_single(x) for x in x_list]
        y_list = y_list[0] if len(y_list) == 1 else y_list
        y_tensor = torch.tensor(y_list)
        return y_tensor

    def __call__(self, x):
        tensor = False
        if isinstance(x, Tensor):
            tensor = True
        y = self.call_tensor(x) if tensor else self.call_single(x)
        return y.to(self.device, self.dtype)

    def to(self, dtype, device):
        self.dtype = dtype
        self.device = device
        return self


if __name__ == "__main__":
    f = SequenceDesignFunction(8).to(device="cuda", dtype=torch.float64)
    # data = np.random.randint(1, 4, size=8)
    samples = torch.randint(0, 3, (5, 8)).to(device="cuda", dtype=torch.float64)
    print(samples)
    print(f(samples))
