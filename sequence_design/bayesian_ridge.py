from typing import (
    Any,
    Optional,
    Union,
)
from botorch.models.utils import validate_input_scaling
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import GPyTorchPosterior
from botorch.posteriors.transformed import TransformedPosterior
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from sklearn.linear_model import BayesianRidge
from torch import Tensor
import torch


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

    def forward(self, x: Tensor) -> Tensor:
        x = self.transform_inputs(x)
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y_mean, y_std = self.br_model.predict(x.cpu().detach().numpy(), return_std=True)

        return MultivariateNormal(
            torch.tensor(y_mean).to(x).reshape(x_shape[:-1]),
            torch.tensor(y_std).to(x).reshape(x_shape[:-1]).unsqueeze(-1)
            * torch.eye(x_shape[-2]).to(x).expand(x_shape[:-1] + (x_shape[-2],)),
        )

    def posterior(
        self,
        X: Tensor,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> Union[GPyTorchPosterior, TransformedPosterior]:

        mvn = self.forward(X)
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


if __name__ == "__main__":
    from sklearn.datasets import fetch_openml

    co2 = fetch_openml(data_id=41187, as_frame=True)
    # co2.frame.head()

    import pandas as pd

    co2_data = co2.frame
    co2_data["date"] = pd.to_datetime(co2_data[["year", "month", "day"]])
    co2_data = co2_data[["date", "co2"]].set_index("date")
    # co2_data.head()

    import matplotlib.pyplot as plt

    # co2_data.plot()
    # plt.ylabel("CO$_2$ concentration (ppm)")
    # _ = plt.title(
    #     "Raw air samples measurements from the Mauna Loa Observatory")

    try:
        co2_data_resampled_monthly = co2_data.resample("ME")
    except ValueError:
        # pandas < 2.2 uses M instead of ME
        co2_data_resampled_monthly = co2_data.resample("M")

    co2_data = co2_data_resampled_monthly.mean().dropna(axis="index", how="any")
    # co2_data.plot()
    # plt.ylabel("Monthly average of CO$_2$ concentration (ppm)")
    # _ = plt.title(
    #     "Monthly average of air samples measurements\nfrom the Mauna Loa Observatory"
    # )

    X = (co2_data.index.year + co2_data.index.month / 12).to_numpy().reshape(-1, 1)
    y = co2_data["co2"].to_numpy().reshape(-1, 1)

    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize

    model = BayesianRidgeModel(
        train_X=torch.tensor(X),
        train_Y=torch.tensor(y),
        input_transform=Normalize(d=1, bounds=torch.tensor([[0], [2050]])),
        # outcome_transform=Standardize(1),
    )
    import datetime
    import numpy as np

    today = datetime.datetime.now()
    current_month = today.year + today.month / 12
    X_test = np.linspace(start=1958, stop=current_month, num=1_000).reshape(-1, 1)
    posterior_predictive = model.posterior(torch.tensor(X_test))
    mean_y_pred = posterior_predictive.mean
    std_y_pred = posterior_predictive.variance

    lcb, ucb = posterior_predictive.distribution.confidence_region()
    mean_y_pred = (ucb + lcb) / 2
    std_y_pred = (ucb - lcb) / 2

    plt.plot(X, y, color="black", linestyle="dashed", label="Measurements")
    plt.plot(
        X_test,
        mean_y_pred,
        color="tab:blue",
        alpha=0.4,
        label="Bayesian Ridge prediction",
    )

    plt.fill_between(
        X_test.ravel(),
        (mean_y_pred - std_y_pred).ravel(),
        (mean_y_pred + std_y_pred).ravel(),
        color="tab:blue",
        alpha=0.2,
    )
    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("Monthly average of CO$_2$ concentration (ppm)")
    plt.title(
        "Monthly average of air samples measurements\nfrom the Mauna Loa Observatory"
    )
    plt.savefig("bayesian_ridge_regression.png")

    # Test condition_on_observations function
    new_model = model.condition_on_observations(
        X=torch.tensor(X),
        Y=torch.tensor(y),
        input_transform=Normalize(d=1, bounds=torch.tensor([[0], [2050]])),
        # outcome_transform=Standardize(1),
    )
