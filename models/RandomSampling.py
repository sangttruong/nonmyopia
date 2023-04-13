import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor


class qRandomSampling(AcquisitionFunction):
    def __init__(self, model, num_samples: int = 64):
        super().__init__(model=model)
        self.num_samples = num_samples

    def forward(self, X: Tensor) -> Tensor:
        """
        Computes the acquisition function value at X.

        Args:
            X: A `batch_shape x q x d`-dim tensor of `q` `d`-dim candidates.

        Returns:
            The acquisition function value at X.
        """
        pass