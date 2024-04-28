import torch
from typing import Any
from tqdm import tqdm


class EnvWrapper:
    def __init__(self, env_name, env):
        self.env = env
        self.bounds = env.bounds
        self.range_y = [self.optimize_min(), self.optimize_max()]
        print("Y range:", self.range_y)
        
        if env_name in ["SynGP", "Alpine"]:
            self.optimal_value = self.range_y[1]
        else:
            self.optimal_value = self.env.optimal_value
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


if __name__ == '__main__':
    from _15_syngp import SynGP
    from _12_alpine import AlpineN1
    from _17_logcos import LogCos
    env = AlpineN1(dim=2)
    # env = SynGP(dim=2)
    # env = LogCos(dim=2)
    env.bounds[0, :] = torch.tensor(
        [1, 0], device=env.bounds.device)
    env.bounds[1, :] = torch.tensor(
        [8, 3], device=env.bounds.device)
    env = EnvWrapper(env)
