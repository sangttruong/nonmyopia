import pickle
import numpy as np
import sys
import statistics
import os

import matplotlib.pyplot as plt

hes_Alpine = [
        "nonmyopia/results/exp_HES_Alpine_11",
        "nonmyopia/results/exp_HES_Alpine_11_00",
        "nonmyopia/results/exp_HES_Alpine_11_01",
    ]

if __name__ == '__main__':
    root = "/dfs/user/sttruong/KhoanWorkspace"
    losses = []
    for idx, path in enumerate(hes_Alpine): 
        full_path = os.path.join(root, path, "metrics.pkl")
        metrics = pickle.load(open(full_path, "rb"))
        print(metrics)
        losses.append(metrics['eval_metric_HES_{}'.format(idx+1)][5:])
    
    losses = np.array(losses)
    mean_hes = np.mean(losses, axis=0)
    std_hes = np.std(losses, axis=0)
    steps = list(range(5, len(losses[0]) + 5))
    print(steps)
    plt.plot(steps, mean_hes, label = "Hes + Alpine")
    plt.fill_between(steps, mean_hes-std_hes, mean_hes+std_hes, alpha=0.1)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.savefig("result.png")