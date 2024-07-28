import unittest
import torch
import random
import botorch
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from botorch.models.transforms.input import Normalize
from botorch.utils import standardize
from botorch.posteriors import GPyTorchPosterior
from torch.distributions.multivariate_normal import MultivariateNormal

from sequence_design.bayesian_ridge import BayesianRidgeModel


class TestBayesianRidde(unittest.TestCase):
    def test_forward(self):
        dim = 8
        rand_coef_ = np.random.rand(dim)
        rand_intercept_ = np.random.rand(1).item()
        rand_sigma_ = np.random.rand(dim, dim)
        rand_alpha_ = (
            np.random.rand(1).item() * np.random.randint(low=0, high=1e4, size=1).item()
        )

        model = BayesianRidgeModel(
            train_X=torch.zeros([1, dim]),
            train_Y=torch.zeros([1, 1]),
        )

        model.br_model.coef_ = rand_coef_
        model.br_model.intercept_ = rand_intercept_
        model.br_model.sigma_ = rand_sigma_
        model.br_model.alpha_ = rand_alpha_

        x_test = np.random.rand(100, dim)
        y_test_dist = model.forward(torch.tensor(x_test))
        y_test_mean = y_test_dist.mean
        y_test_std = y_test_dist.covariance_matrix.diag()

        y_test_mean_truth = x_test @ rand_coef_ + rand_intercept_
        sigmas_squared_data = (np.dot(x_test, rand_sigma_) * x_test).sum(axis=1)
        y_test_std_truth = np.sqrt(sigmas_squared_data + (1.0 / rand_alpha_))

        assert (y_test_mean.numpy() == y_test_mean_truth).all()
        assert (y_test_std.numpy() == y_test_std_truth).all()

    def test_posterior(self):
        dim = 8

        model = BayesianRidgeModel(
            train_X=torch.rand([100, dim]),
            train_Y=torch.rand([100, 1]),
        )
        x_test = np.random.rand(100, dim)
        y_test_dist = model.forward(torch.tensor(x_test))

        y_test_posterior = model.posterior(torch.tensor(x_test))

        assert isinstance(y_test_posterior, GPyTorchPosterior)
        assert (y_test_posterior.mvn.mean == y_test_dist.mean).all()
        assert (y_test_posterior.mvn.stddev == y_test_dist.stddev).all()

    def test_regression(self):
        co2 = fetch_openml(data_id=41187, as_frame=True)
        co2_data = co2.frame
        co2_data["date"] = pd.to_datetime(co2_data[["year", "month", "day"]])
        co2_data = co2_data[["date", "co2"]].set_index("date")
        try:
            co2_data_resampled_monthly = co2_data.resample("ME")
        except ValueError:
            # pandas < 2.2 uses M instead of ME
            co2_data_resampled_monthly = co2_data.resample("M")

        co2_data = co2_data_resampled_monthly.mean().dropna(axis="index", how="any")

        X = (co2_data.index.year + co2_data.index.month / 12).to_numpy().reshape(-1, 1)
        y = co2_data["co2"].to_numpy().reshape(-1, 1)

        X_train = X[:450]
        y_train = y[:450]
        X_test = X[450:]
        y_test = y[450:]

        model = BayesianRidgeModel(
            train_X=torch.tensor(X_train),
            train_Y=torch.tensor(y_train),
            input_transform=Normalize(d=1, bounds=torch.tensor([[0], [2050]])),
            # outcome_transform=Standardize(1),
        )

        posterior_predictive = model.posterior(torch.tensor(X_test))
        mean_y_pred = posterior_predictive.mean

        assert (
            torch.nn.functional.l1_loss(mean_y_pred, torch.tensor(y_test)).item() < 4.0
        )

    def test_condition(self):
        co2 = fetch_openml(data_id=41187, as_frame=True)
        co2_data = co2.frame
        co2_data["date"] = pd.to_datetime(co2_data[["year", "month", "day"]])
        co2_data = co2_data[["date", "co2"]].set_index("date")
        try:
            co2_data_resampled_monthly = co2_data.resample("ME")
        except ValueError:
            # pandas < 2.2 uses M instead of ME
            co2_data_resampled_monthly = co2_data.resample("M")

        co2_data = co2_data_resampled_monthly.mean().dropna(axis="index", how="any")

        X = (co2_data.index.year + co2_data.index.month / 12).to_numpy().reshape(-1, 1)
        y = co2_data["co2"].to_numpy().reshape(-1, 1)

        X_train = X[:450]
        y_train = y[:450]
        X_test = X[450:]
        y_test = y[450:]

        model = BayesianRidgeModel(
            train_X=torch.tensor(X_train),
            train_Y=torch.tensor(y_train),
            input_transform=Normalize(d=1, bounds=torch.tensor([[0], [2050]])),
            # outcome_transform=Standardize(1),
        )

        new_model = model.condition_on_observations(
            X=torch.tensor(X_test),
            Y=torch.tensor(y_test),
            input_transform=Normalize(d=1, bounds=torch.tensor([[0], [2050]])),
            # outcome_transform=Standardize(1),
        )

        assert (
            torch.nn.functional.l1_loss(
                new_model.posterior(torch.tensor(X_test)).mean, torch.tensor(y_test)
            )
            < 3.0
        )

    def test_aquisition_function(self):
        seed = 3
        torch.manual_seed(seed=seed)
        np.random.seed(seed)
        random.seed(seed)
        botorch.utils.sampling.manual_seed(seed=seed)

        train_X = torch.rand(10, 2)
        Y = 1 - torch.norm(train_X - 0.5, dim=-1, keepdim=True)
        Y = Y + 0.1 * torch.randn_like(Y)  # add some noise
        train_Y = standardize(Y)

        test_X = torch.rand(1, 1, 2)

        model = BayesianRidgeModel(train_X, train_Y)

        best_f = train_Y.max()
        sampler = botorch.sampling.SobolQMCNormalSampler(10000)

        qEI = botorch.acquisition.monte_carlo.qExpectedImprovement(
            model, best_f, sampler
        )
        qei = qEI(test_X)

        qPi = botorch.acquisition.monte_carlo.qProbabilityOfImprovement(
            model, best_f, sampler
        )
        qpi = qPi(test_X)
        dist = MultivariateNormal(
            model.posterior(test_X).mean, model.posterior(test_X).variance
        )

        samples = [torch.relu(dist.sample() - best_f) for _ in range(10000)]
        mean_ei = torch.mean(torch.stack(samples), dim=0)

        samples = [torch.sigmoid((dist.sample() - best_f) / 1e-3) for _ in range(10000)]
        mean_pi = torch.mean(torch.stack(samples), dim=0)

        assert abs(qei - mean_ei) < 0.5
        assert abs(qpi - mean_pi) < 0.5


if __name__ == "__main__":
    unittest.main()
