import random
import torch
import os
import csv
import pickle

import numpy as np

from botorch import test_functions
from embedder import DiscreteEmbbeder

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIM = 10


def write_file(path_save, data):
    header = ["chain", "chain_index", "sample", "score"]
    with open(path_save, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        for line in data:
            chain_index = " ".join([str(idx.item()) for idx in line[1]])
            sample = " ".join([str(idx.item()) for idx in line[2]])
            score = line[-1].item()
            writer.writerow([line[0], chain_index, sample, score])


def generate(save_folder, ackley_function, number_samples, training_ratio):
    embedder = DiscreteEmbbeder(
        num_categories=len(ALPHABET),
        bounds=(-1.0, 1.0),
    ).to(device="cuda", dtype=torch.float64)

    datas = []
    while len(datas) < number_samples:
        sample = torch.tensor(np.random.uniform(-1, 1, DIM)).to(
            device="cuda", dtype=torch.float64
        )
        chain_index = embedder.decode(sample)
        chain = "".join(ALPHABET[i] for i in chain_index)
        score = ackley_function(sample)
        datas.append([chain, chain_index, sample, score])

    training_sample = int(number_samples * training_ratio)
    training_data = datas[:training_sample]
    testing_data = datas[training_sample:]

    write_file(os.path.join(save_folder, "train.csv"), training_data)
    write_file(os.path.join(save_folder, "test.csv"), testing_data)


if __name__ == "__main__":
    ackley_function = test_functions.synthetic.Ackley(
        dim=DIM, bounds=[(-1.0, 1.0)] * DIM
    )

    generate(
        save_folder="./data",
        ackley_function=ackley_function,
        number_samples=5000,
        training_ratio=0.7,
    )
