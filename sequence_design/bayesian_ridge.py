from typing import Any, Optional, Union

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.posteriors import GPyTorchPosterior
from botorch.posteriors.transformed import TransformedPosterior
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from sklearn.linear_model import BayesianRidge
from torch import Tensor


class BayesianRidgeModel(BatchedMultiOutputGPyTorchModel):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        super(BayesianRidgeModel, self).__init__()

        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        self._validate_tensor_args(X=transformed_X, Y=train_Y, Yvar=train_Yvar)
        ignore_X_dims = getattr(self, "_ignore_X_dims_scaling_check", None)
        validate_input_scaling(
            train_X=transformed_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            ignore_X_dims=ignore_X_dims,
        )
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, train_Yvar = self._transform_tensor_args(
            X=train_X, Y=train_Y, Yvar=train_Yvar
        )

        self.train_inputs = train_X
        self.train_targets = train_Y

        self.br_model = BayesianRidge()
        self.br_model.fit(X=transformed_X, y=train_Y)

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

    def forward(self, X: Tensor) -> Tensor:
        X = self.transform_inputs(X)
        X_shape = X.shape
        X = X.view(-1, X_shape[-1])
        y_mean, y_std = self.br_model.predict(X.cpu().detach().numpy(), return_std=True)

        return MultivariateNormal(
            torch.tensor(y_mean).to(X).reshape(X_shape[:-1]),
            torch.tensor(y_std).to(X).reshape(X_shape[:-1]).unsqueeze(-1)
            * torch.eye(X_shape[-2]).to(X).expand(X_shape[:-1] + (X_shape[-2],)),
        )

    def predict(self, X: Tensor) -> Tensor:
        """
        This functin return the mean outputs.
        """
        X = self.transform_inputs(X)
        X_shape = X.shape
        X = X.view(-1, X_shape[-1])
        y_mean = self.br_model.predict(X.cpu().detach().numpy())
        return torch.tensor(y_mean).to(X)

    def sample(self, X: Tensor, sample_size: int = 1, **kwargs):
        """
        This functin return sampled outputs.
        """
        posteriors = self.posterior(X=X, **kwargs)
        posterior_pred = posteriors.sample(
            sample_shape=torch.Size([sample_size])
        ).squeeze(-1)
        return posterior_pred

    def posterior(
        self,
        X: Tensor,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> Union[GPyTorchPosterior, TransformedPosterior]:

        mvn = self.forward(X=X)
        posterior = GPyTorchPosterior(distribution=mvn)

        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    def get_fantasy_model(
        self, inputs: Tensor, targets: Tensor, **kwargs: Any
    ) -> Model:
        r"""Create a new model with the specified fantasy observations.

        Args:
            inputs: A `batch_shape x n x d`-dim Tensor, where `d` is the dimension of
                the feature space, and `n` is the number of fantasy points.
            targets: A `batch_shape x n x m`-dim Tensor, where `m` is the number of
                model outputs.
            **kwargs: Additional arguments, passed to the constructor of the new model.

        Returns:
            A new model, created by adding the fantasy observations to the training data
            of the current model.
        """
        new_inputs = torch.concatenate([self.train_inputs, inputs])
        new_targets = torch.concatenate([self.train_targets, targets]).reshape(-1, 1)
        fantasized_model = BayesianRidgeModel(
            train_X=new_inputs, train_Y=new_targets, **kwargs
        )
        return fantasized_model
