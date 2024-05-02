import warnings
from abc import ABC
from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Union,
)
from botorch.exceptions.errors import (
    BotorchTensorDimensionError,
    InputDataError,
)
from botorch.exceptions.warnings import (
    _get_single_precision_warning,
    BotorchTensorDimensionWarning,
    InputDataWarning,
)
from botorch.models.utils import (
    # gpt_posterior_settings,
    multioutput_to_batch_mode_transform,
    validate_input_scaling
)
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import TorchPosterior
from botorch.posteriors.transformed import TransformedPosterior
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from sklearn.linear_model import BayesianRidge
from torch import Tensor
import torch


class BayesianRidgeModel(Model, ABC):
    @staticmethod
    def _validate_tensor_args(
        X: Tensor, Y: Tensor, Yvar: Optional[Tensor] = None, strict: bool = True
    ) -> None:
        r"""Checks that `Y` and `Yvar` have an explicit output dimension if strict.
        Checks that the dtypes of the inputs match, and warns if using float.

        This also checks that `Yvar` has the same trailing dimensions as `Y`. Note
        we only infer that an explicit output dimension exists when `X` and `Y` have
        the same `batch_shape`.

        Args:
            X: A `batch_shape x n x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n` is the number of points per batch, and
                `batch_shape` is the batch shape (potentially empty).
            Y: A `batch_shape' x n x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
            Yvar: A `batch_shape' x n x m` tensor of observed measurement noise.
                Note: this will be None when using a model that infers the noise
                level (e.g. a `SingleTaskGP`).
            strict: A boolean indicating whether to check that `Y` and `Yvar`
                have an explicit output dimension.
        """
        if X.dim() != Y.dim():
            if (X.dim() - Y.dim() == 1) and (X.shape[:-1] == Y.shape):
                message = (
                    "An explicit output dimension is required for targets."
                    f" Expected Y with dimension {X.dim()} (got {Y.dim()=})."
                )
            else:
                message = (
                    "Expected X and Y to have the same number of dimensions"
                    f" (got X with dimension {X.dim()} and Y with dimension"
                    f" {Y.dim()})."
                )
            if strict:
                raise BotorchTensorDimensionError(message)
            else:
                warnings.warn(
                    "Non-strict enforcement of botorch tensor conventions. The "
                    "following error would have been raised with strict enforcement: "
                    f"{message}",
                    BotorchTensorDimensionWarning,
                    stacklevel=2,
                )
        # Yvar may not have the same batch dimensions, but the trailing dimensions
        # of Yvar should be the same as the trailing dimensions of Y.
        if Yvar is not None and Y.shape[-(Yvar.dim()):] != Yvar.shape:
            raise BotorchTensorDimensionError(
                "An explicit output dimension is required for observation noise."
                f" Expected Yvar with shape: {Y.shape[-Yvar.dim() :]} (got"
                f" {Yvar.shape})."
            )
        # Check the dtypes.
        if X.dtype != Y.dtype or (Yvar is not None and Y.dtype != Yvar.dtype):
            raise InputDataError(
                "Expected all inputs to share the same dtype. Got "
                f"{X.dtype} for X, {Y.dtype} for Y, and "
                f"{Yvar.dtype if Yvar is not None else None} for Yvar."
            )
        if X.dtype != torch.float64:
            warnings.warn(
                _get_single_precision_warning(str(X.dtype)),
                InputDataWarning,
                stacklevel=3,  # Warn at model constructor call.
            )

    def _transform_tensor_args(
        self, X: Tensor, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        r"""Transforms tensor arguments: for single output models, the output
        dimension is squeezed and for multi-output models, the output dimension is
        transformed into the left-most batch dimension.

        Args:
            X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                training observations.
            Yvar: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                observed measurement noise. Note: this will be None when using a model
                that infers the noise level (e.g. a `SingleTaskGP`).

        Returns:
            3-element tuple containing

            - A `input_batch_shape x (m) x n x d` tensor of training features.
            - A `target_batch_shape x (m) x n` tensor of training observations.
            - A `target_batch_shape x (m) x n` tensor observed measurement noise
                (or None).
        """
        if self._num_outputs > 1:
            return multioutput_to_batch_mode_transform(
                train_X=X, train_Y=Y, train_Yvar=Yvar, num_outputs=self._num_outputs
            )
        return X, Y.squeeze(-1), None if Yvar is None else Yvar.squeeze(-1)

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

        self.br_model = BayesianRidge()
        self.br_model.fit(X=transformed_X, y=train_Y)

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        # self.to(train_X)

    def _set_dimensions(self, train_X: Tensor, train_Y: Tensor) -> None:
        r"""Store the number of outputs and the batch shape.

        Args:
            train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                training observations.
        """
        self._num_outputs = train_Y.shape[-1]
        self._input_batch_shape, self._aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )

    @staticmethod
    def get_batch_dimensions(
        train_X: Tensor, train_Y: Tensor
    ) -> Tuple[torch.Size, torch.Size]:
        r"""Get the raw batch shape and output-augmented batch shape of the inputs.

        Args:
            train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                training observations.

        Returns:
            2-element tuple containing

            - The `input_batch_shape`
            - The output-augmented batch shape: `input_batch_shape x (m)`
        """
        input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = input_batch_shape
        num_outputs = train_Y.shape[-1]
        if num_outputs > 1:
            aug_batch_shape += torch.Size([num_outputs])
        return input_batch_shape, aug_batch_shape

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        raise NotImplementedError

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return self._num_outputs

    def subset_output(self, idcs: List[int]) -> Model:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            A `Model` object of the same type and with the same parameters as
            the current model, subset to the specified output indices.
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        x = self.transform_inputs(x)
        y_mean, y_std = self.br_model.predict(x.cpu().numpy(), return_std=True)
        return MultivariateNormal(
            torch.tensor(y_mean).to(x).unsqueeze(-1),
            torch.tensor(y_std).to(x).reshape((-1, 1, 1)),
        )
        # return MultivariateNormal(
        #     torch.tensor(y_mean).to(x).unsqueeze(0),
        #     torch.diag(torch.tensor(y_std).to(x), diagonal=0).unsqueeze(0),
        # )

    def posterior(
        self,
        X: Tensor,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> Union[TorchPosterior, TransformedPosterior]:

        mvn = self.forward(X)
        posterior = TorchPosterior(distribution=mvn)

        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        r"""Condition the model on new observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n' x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, it is assumed that the missing batch dimensions are
                the same for all `Y`.

        Returns:
            A `Model` object of the same type, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).
        """
        Yvar = kwargs.pop("noise", None)

        if hasattr(self, "outcome_transform"):
            # pass the transformed data to get_fantasy_model below
            # (unless we've already trasnformed if BatchedMultiOutputGPyTorchModel)
            # `noise` is assumed to already be outcome-transformed.
            Y, _ = self.outcome_transform(Y, Yvar)
        # validate using strict=False, since we cannot tell if Y has an explicit
        # output dimension
        self._validate_tensor_args(X=X, Y=Y, Yvar=Yvar, strict=False)
        if Y.size(-1) == 1:
            Y = Y.squeeze(-1)
            if Yvar is not None:
                kwargs.update({"noise": Yvar.squeeze(-1)})
        # get_fantasy_model will properly copy any existing outcome transforms
        # (since it deepcopies the original model)

        return self.get_fantasy_model(inputs=X, targets=Y, **kwargs)

    def get_fantasy_model(self, inputs: Tensor, targets: Tensor, **kwargs: Any) -> Model:
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
        fantasized_model = BayesianRidgeModel(
            train_X=inputs,  train_Y=targets, **kwargs)
        return fantasized_model


if __name__ == '__main__':
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
    X_test = np.linspace(start=1958, stop=current_month,
                         num=1_000).reshape(-1, 1)
    posterior_predictive = model.posterior(torch.tensor(X_test))
    mean_y_pred = posterior_predictive.mean
    std_y_pred = posterior_predictive.variance

    lcb, ucb = posterior_predictive.distribution.confidence_region()
    mean_y_pred = (ucb + lcb) / 2
    std_y_pred = (ucb - lcb) / 2

    plt.plot(X, y, color="black", linestyle="dashed", label="Measurements")
    plt.plot(X_test, mean_y_pred, color="tab:blue",
             alpha=0.4, label="Bayesian Ridge prediction")

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
