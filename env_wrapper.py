import torch
from typing import Any
from tqdm import tqdm


class EnvWrapper:
    def __init__(self, env_name, env):
        self.env = env
        self.bounds = env.bounds
        
        if env_name in ["SynGP", "Alpine"]:
            self.optimal_value = self.optimize_max()
        else:
            self.optimal_value = self.env.optimal_value
            
        self.range_y = [self.optimize_min(), self.optimal_value]
        print("Y range:", self.range_y)
        print("Optimal value:", self.optimal_value)

    def optimize_min(self):
        def _min_fn_():
            # Sample 10000 points and find the minimum
            inputs = torch.rand((10000, self.env.dim))
            inputs = inputs * (self.env.bounds[1] -
                               self.env.bounds[0]) + self.env.bounds[0]
            res = self.env(inputs)
            return res.min().item()

        min_val = _min_fn_()
        for _ in tqdm(range(9), desc="Optimizing min"):
            min_val = min(min_val, _min_fn_())
        return min_val

    def optimize_max(self):
        def _max_fn_():
            # Sample 10000 points and find the maximum
            inputs = torch.rand((10000, self.env.dim))
            inputs = inputs * (self.env.bounds[1] -
                               self.env.bounds[0]) + self.env.bounds[0]
            res = self.env(inputs)
            return res.max().item()

        max_val = _max_fn_()
        for _ in tqdm(range(9), desc="Optimizing max"):
            max_val = max(max_val, _max_fn_())
        return max_val

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        res = self.env(*args, **kwds)

        # Normalize output
        res = (res - self.range_y[0]) / (self.range_y[1] - self.range_y[0])
        return res

    def to(self, dtype, device):
        self.env = self.env.to(dtype=dtype, device=device)
        return self
