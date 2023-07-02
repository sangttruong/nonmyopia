import random
import torch
import os
import csv
import pickle

import pandas as pd
import numpy as np

from botorch import test_functions
from generate_data import ALPHABET, DIM
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch.nn as nn

SEED = 42

def load_data(path):
    data = pd.read_csv(path)
    samples = []
    for sample in data.iloc[:, 2].values:
        sample_tensor = torch.zeros(DIM).to(device="cuda", dtype=torch.float64)
        for idx, s in enumerate(sample.split()):
            sample_tensor[idx] = float(s)
        samples.append(sample_tensor)
        
    samples = torch.stack(samples)
    scores = torch.tensor(
        [[score] for score in data.iloc[:, -1].values]
    ).to(device="cuda", dtype=torch.float64)
    return samples, scores


def write_result(path_save, y_predict, y_ground_truth, loss):
    header = ["predict", "groundtruth"]
    with open(path_save, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(header)
        assert len(y_predict) == len(y_ground_truth)
        for p, g in zip(y_predict, y_ground_truth):
            writer.writerow([p.item(), g.item()])
        writer.writerow(['Loss', loss.item()])


def train(x_train, y_train):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    WM = SingleTaskGP(
        x_train,
        y_train,
    ).to(device="cuda")
    
    mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
    fit_gpytorch_model(mll)
    
    return WM
    
    
def test(x_test, y_test, model):
    with torch.no_grad():
        posterior = model.posterior(x_test)
        loss = nn.MSELoss()
        output = loss(posterior.mean, y_test)
        write_result("result.csv", posterior.mean, y_test, output)


def main():
    x_train, y_train = load_data(os.path.join("./data", 'train.csv'))
    x_test, y_test = load_data(os.path.join("./data", 'test.csv'))

    model = train(x_train=x_train, y_train=y_train)    
    
    test(x_test=x_test, y_test=y_test, model=model) 
    
    
if __name__ == '__main__':
    main()
