import random
import torch
import os
import csv
import math

import pandas as pd
import numpy as np
import torch.nn as nn

from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm
from kernel import TransformedCategorical

SEED = 42
DIM = 8
ALPHABET = "ABC"


def sequence_score(sequence):
    number_b = sequence.count("B")
    number_c = sequence.count("C")
    threshold = int(math.ceil(len(sequence) / 2))
    if number_c == DIM:
        return len(sequence) + 1

    if number_c >= threshold:
        score_c = ((len(sequence) + 1 + threshold)) / ((len(sequence) - threshold))
        return number_b - threshold + (number_c % threshold) * (score_c)
    else:
        return number_b - number_c


def generate_sequence(length):
    results = []

    def generate_binary_sequence(n, seq=""):
        if len(seq) == n:
            results.append(seq)
        else:
            generate_binary_sequence(n, seq + "A")
            generate_binary_sequence(n, seq + "B")
            generate_binary_sequence(n, seq + "C")

    generate_binary_sequence(length)
    return results


def load_data(is_train=True, number_sample=5000):
    if is_train:
        samples = torch.randint(1, 4, (number_sample, DIM)).to(
            device="cuda", dtype=torch.float64
        )
        scores = []
        for idxs in tqdm(samples):
            sequence = ""
            for idx in idxs:
                sequence += ALPHABET[int(idx.item()) - 1]
            score = sequence_score(sequence=sequence)
            scores.append([score])
        scores = torch.tensor(scores).to(device="cuda", dtype=torch.float64)
        return samples, scores
    else:
        data_path = "./data/test_{}.csv".format(DIM)
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            samples = []
            for sample in tqdm(data.iloc[:, 1].values):
                sample_tensor = torch.zeros(DIM)
                for idx, s in enumerate(sample.split()):
                    sample_tensor[idx] = float(s)
                samples.append(sample_tensor)

            samples = torch.stack(samples).to(device="cuda", dtype=torch.float64)
            scores = torch.tensor([[score] for score in data.iloc[:, -1].values]).to(
                device="cuda", dtype=torch.float64
            )
            return samples, scores
        else:
            samples = []
            scores = []
            datas = []
            sequences = generate_sequence(DIM)
            for sequence in tqdm(sequences):
                sample = np.zeros(DIM)
                for idx, c in enumerate(sequence):
                    if c == "A":
                        sample[idx] = 1
                    elif c == "B":
                        sample[idx] = 2
                    else:
                        sample[idx] = 3
                sample = torch.tensor(sample).to(device="cuda", dtype=torch.float64)
                score = sequence_score(sequence=sequence)
                datas.append([sequence, sample, score])
                samples.append(sample)
                scores.append([score])

            write_file(data_path, datas)

            samples = torch.stack(samples)
            scores = torch.tensor(scores).to(device="cuda", dtype=torch.float64)
            return samples, scores


def write_result(path_save, y_predict, y_ground_truth, loss):
    header = ["predict", "groundtruth"]
    with open(path_save, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(header)
        assert len(y_predict) == len(y_ground_truth)
        for p, g in zip(y_predict, y_ground_truth):
            writer.writerow([p.item(), g.item()])
        writer.writerow(["Loss", loss.item()])


def train(x_train, y_train):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    kernel = TransformedCategorical()

    WM = SingleTaskGP(
        x_train,
        y_train,
        covar_module=kernel,
    ).to(device="cuda")

    mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
    fit_gpytorch_model(mll)

    return WM


def test(x_test, y_test, model):
    with torch.no_grad():
        posterior = model.posterior(x_test)
        loss = nn.MSELoss()
        output = loss(posterior.mean, y_test)
        write_result(
            "./result/result_{}.csv".format(DIM), posterior.mean, y_test, output
        )


def write_file(path_save, data):
    header = ["chain", "sample", "score"]
    with open(path_save, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        for line in data:
            sample = " ".join([str(idx.item()) for idx in torch.flatten(line[1])])
            writer.writerow([line[0], sample, line[-1]])


def main():
    x_train, y_train = load_data(is_train=True, number_sample=5000)
    x_test, y_test = load_data(is_train=False)
    model = train(x_train=x_train, y_train=y_train)

    test(x_test=x_test, y_test=y_test, model=model)


if __name__ == "__main__":
    main()
