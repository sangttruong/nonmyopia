from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import botorch
import random
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np

from bayesian_ridge import BayesianRidgeModel

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

# model = SingleTaskGP(train_X, train_Y)
# mll = ExactMarginalLogLikelihood(model.likelihood, model)
# fit_gpytorch_model(mll)

model = BayesianRidgeModel(train_X, train_Y)

best_f = train_Y.max()
sampler = botorch.sampling.SobolQMCNormalSampler(10000)

qEI = botorch.acquisition.monte_carlo.qExpectedImprovement(model, best_f, sampler)
qei = qEI(test_X)
print("qEI:", qei)

qPi = botorch.acquisition.monte_carlo.qProbabilityOfImprovement(model, best_f, sampler)
qpi = qPi(test_X)
print("qPI:", qpi)

print("Test X:", test_X)
print("Posterior mean:", model.posterior(test_X).mean)
print("Posterior variance:", model.posterior(test_X).variance)


dist = MultivariateNormal(
    model.posterior(test_X).mean, model.posterior(test_X).variance
)

samples = [torch.relu(dist.sample() - best_f) for _ in range(10000)]
mean = torch.mean(torch.stack(samples), dim=0)
print("MC-EI:", mean)


samples = [torch.sigmoid((dist.sample() - best_f) / 1e-3) for _ in range(10000)]
mean = torch.mean(torch.stack(samples), dim=0)
print("MC-PI:", mean)
