import os

import pandas as pd
import matplotlib.pyplot as plt

from train import DIM
from tqdm import tqdm


def get_data():
    data_path = "./result/result_{}.csv".format(DIM)
    samples = []
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        for idx, sample in tqdm(data.iterrows()):
            if idx == len(data) - 1:
                break
            samples.append((float(sample[0]), float(sample[1])))
    return list(zip(*samples))


def main():
    data = get_data()
    plt.plot(data[0], label="Predicted value")
    plt.plot(data[1], label="Real value")
    plt.legend(loc="upper left")
    plt.savefig("result.png")


if __name__ == "__main__":
    main()
