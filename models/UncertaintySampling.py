import torch
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor


class qUncertaintySampling(AcquisitionFunction):
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
        # Draw samples from the posterior
        posterior_samples = self.model.posterior(X).rsample(
            torch.Size([self.num_samples])
        )
        # Compute the expected improvement of the samples
        expected_improvement = qExpectedImprovement(
            self.model, self.model.train_targets.min(), maximize=False
        )(X)
        # Compute the variance of the posterior samples
        variance = posterior_samples.var(dim=0)
        # Compute the uncertainty score as the variance of the posterior samples times the expected improvement
        return variance * expected_improvement
