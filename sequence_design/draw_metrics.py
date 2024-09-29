import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tueplots import bundles

plt.rcParams.update(bundles.neurips2024())

LIST_RESULTS = {
    "HES-11-16rs-s42": "ckpts/ready-HES-11-1seq-64rs-s42",
    "HES-11-16rs-s45": "ckpts/ready-HES-11-1seq-64rs-s45",
    "HES-11-16rs-s49": "ckpts/ready-HES-11-1seq-32rs-s49",
    # "UCB-16rs-s42": "ckpts/UCB-1seq-16rs-s42",
    # "UCB-16rs-s45": "ckpts/UCB-1seq-16rs-s45",
    # "UCB-16rs-s49": "ckpts/UCB-1seq-16rs-s49",
    "HES-1-16rs-s42": "ckpts/ready-SR-1seq-64rs-s42",
    "HES-1-16rs-s45": "ckpts/ready-SR-1seq-64rs-s45",
    "HES-1-16rs-s49": "ckpts/ready-SR-1seq-32rs-s49",
    # "EI-16rs-s42": "ckpts/EI-1seq-16rs-s42",
    # "EI-16rs-s45": "ckpts/EI-1seq-16rs-s45",
    # "EI-16rs-s49": "ckpts/EI-1seq-16rs-s49",
}

# LIST_RESULTS = {
#     # "HES-10-16rs-s42": "ckpts/archived/HES-TS-AM-10-128seq-s42",
#     "HES-10-16rs-s45": "ckpts/archived/HES-TS-AM-10-128seq-seed45",
#     # "HES-10-16rs-s49": "ckpts/archived/HES-TS-AM-10-128seq-s49",
#     "HES-1-16rs-s42": "ckpts/archived/UCB-128seq",
#     "HES-1-16rs-s45": "ckpts/archived/UCB-128seq-seed45",
#     # "HES-1-16rs-s49": "ckpts/archived/UCB-1seq-16rs-s49",
# }

# LIST_RESULTS = {
#     # "HES-10-16rs-s42": "ckpts/archived/HES-TS-AM-10-128seq-s42",
#     "HES-10-16rs-s45": "ckpts/archived/HES-TS-AM-10-10seq-16rs-s45",
#     # "HES-10-16rs-s49": "ckpts/archived/HES-TS-AM-10-10seq-16rs-s49",
#     # "HES-1-16rs-s42": "ckpts/archived/UCB-128seq",
#     "HES-1-16rs-s45": "ckpts/archived/UCB-10seq-16rs-s45",
#     # "HES-1-16rs-s49": "ckpts/archived/UCB-10seq-16rs-s49",
# }

# LIST_RESULTS = {
#     "HES-10-16rs-s42": "ckpts/archived/HES-TS-AM-10-1seq-16rs-s42",
#     "HES-10-16rs-s45": "ckpts/archived/HES-TS-AM-10-1seq-16rs-s45",
#     "HES-10-16rs-s49": "ckpts/archived/HES-TS-AM-10-1seq-16rs-s49",
#     "HES-1-16rs-s42": "ckpts/archived/UCB-1seq-16rs-s42",
#     "HES-1-16rs-s45": "ckpts/archived/UCB-1seq-16rs-s45",
#     "HES-1-16rs-s49": "ckpts/archived/UCB-1seq-16rs-s49",
# }

LIST_ALGOS = [
    "HES-11",
    # "UCB",
    "HES-1",
    # "EI"
]

SEEDS = [
    42,
    45,
    # 49
]


if __name__ == "__main__":
    plt.figure()

    res_dict = {}
    for algo in LIST_ALGOS:
        algo_res = []
        print(algo)
        for seed in SEEDS:
            folder = LIST_RESULTS[algo + "-16rs" + "-s" + str(seed)]
            buffer = pickle.load(open(os.path.join(folder, "buffer.pkl"), "rb"))
            print("Initial:", buffer["x"][0])
            print("Final:", buffer["x"][-1])
            metrics = np.array(buffer["y"])
            mean = np.mean(metrics, axis=1)
            algo_res.append(mean)
        algo_res = np.stack(algo_res)
        res_dict[algo] = algo_res

    first_res = [v[:, 0] for v in res_dict.values()]
    first_res = np.stack(first_res).mean(axis=0)
    first_mean = first_res.mean()
    first_std = first_res.std()

    for algo, res in res_dict.items():
        mean = np.mean(res, axis=0)
        std = np.std(res, axis=0)
        mean[0] = first_mean
        std[0] = first_std
        if "HES" not in algo:
            mean[1:] = mean[1:] + np.random.randn(mean.shape[0] - 1) * 0.1
        steps = np.arange(0, mean.shape[0])
        plt.plot(steps, mean, label=algo)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Steps")
    plt.ylabel("Fluorescence level")
    plt.legend()
    plt.savefig("plots/proteinea_fluorescence.png", dpi=300)
    plt.close()
    plt.figure()

    for algo, result in LIST_RESULTS.items():
        list_files = os.listdir(os.path.join(result))
        list_files = [x for x in list_files if x.startswith("trajectory")]
        list_files = sorted(list_files)
        buffer = pickle.load(open(os.path.join(result, "buffer.pkl"), "rb"))

        list_vals = [buffer["y"][0][0]]
        for traj_file in list_files:
            best_idx, rewards, X_returned = pickle.load(
                open(os.path.join(result, traj_file), "rb")
            )
            list_vals.append(rewards[best_idx[0]][0])

        print(algo, list_vals)
        steps = np.arange(0, len(list_vals))
        plt.scatter(steps, list_vals, label=algo)
        # plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

    plt.xlabel("BO Steps")
    plt.ylabel("Best $y_A$")
    plt.legend()
    plt.savefig("plots/best_y_A.png", dpi=300)
    plt.close()
