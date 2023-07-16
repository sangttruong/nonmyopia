import random
import torch
import os
import csv
import math

import numpy as np

from tqdm import tqdm

ALPHABET = "ABC"
DIM = 10


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


def generate(save_folder, training_ratio):
    datas = []
    sequences = generate_sequence(DIM)
    for sequence in tqdm(sequences):
        sample = np.zeros((DIM, 3))
        for idx, c in enumerate(sequence):
            if c == "A":
                sample[idx][0] = 1
            elif c == "B":
                sample[idx][1] = 1
            else:
                sample[idx][2] = 1
        sample = torch.tensor(sample).to(device="cuda", dtype=torch.float64)
        score = sequence_score(sequence=sequence)
        datas.append([sequence, sample, score])

    number_samples = len(sequences)
    training_sample = int(number_samples * training_ratio)
    random.shuffle(datas)
    training_data = datas[:training_sample]
    testing_data = datas[training_sample:]

    write_file(os.path.join(save_folder, "train.csv"), training_data)
    write_file(os.path.join(save_folder, "test.csv"), testing_data)


if __name__ == "__main__":
    generate(save_folder="./data", training_ratio=0.7)
