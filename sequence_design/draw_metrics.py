import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tueplots import bundles

plt.rcParams.update(bundles.neurips2024())

LIST_RESULTS = {"HES-TS-AM-10": "ckpts/HES-TS-AM-10-10seq"}

if __name__ == "__main__":
    plt.figure()

    for algo, result in LIST_RESULTS.items():
        buffer = pickle.load(open(os.path.join(result, "buffer.pkl"), "rb"))
        metrics = np.array(buffer["y"])
        mean = np.mean(metrics, axis=1)
        std = np.std(metrics, axis=1)
        steps = np.arange(0, mean.shape[0])
        plt.plot(steps, mean, label=algo)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Steps")
    plt.ylabel("Fluorescence level")
    plt.legend()
    plt.savefig("plots/proteinea_fluorescence.png", dpi=300)
    plt.close()
