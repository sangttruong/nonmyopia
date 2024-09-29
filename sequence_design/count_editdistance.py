import torch
from datasets import load_dataset

ds = load_dataset("stair-lab/proteinea_fluorescence-gemma-7b-embedding")

reward_train = torch.tensor(ds["train"]["reward"])
reward_idx = reward_train.argmax()

print(ds["train"]["text"][reward_idx])
print(ds["train"]["reward"][reward_idx])


reward_train = torch.tensor(ds["test"]["reward"])
reward_idx = reward_train.argmax()

print(ds["test"]["text"][reward_idx])
print(ds["test"]["reward"][reward_idx])

from time import time

import numpy as np
from Levenshtein import distance
from tqdm import tqdm


def get_distance_matrix(str_list):
    """Construct a levenshtein distance matrix for a list of strings"""
    dist_matrix = np.zeros(shape=(len(str_list), len(str_list)))
    t0 = time()
    print(
        "Starting to build distance matrix. This will iterate from 0 till ",
        len(str_list),
    )
    for i in tqdm(range(0, len(str_list))):
        for j in range(i + 1, len(str_list)):
            dist_matrix[i][j] = distance(str_list[i], str_list[j])
    for i in range(0, len(str_list)):
        for j in range(0, len(str_list)):
            if i == j:
                dist_matrix[i][j] = 0
            elif i > j:
                dist_matrix[i][j] = dist_matrix[j][i]
    t1 = time()
    return dist_matrix


dist_mat = get_distance_matrix(ds["train"]["text"])

import pickle

pickle.dump(dist_mat, open("dist_mat.pkl", "wb"))
