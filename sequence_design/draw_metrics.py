import argparse
import os
import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import ScalarFormatter
from tueplots import bundles, figsizes
from utils import (
    compute_ed,
    ensure_dir,
    get_embedding_from_server,
    import_protein_env,
    observe_value,
)

plt.rcParams.update(bundles.iclr2024())

# M1F1
# LIST_RESULTS = {
#     "SR-s42": "ckpts_iclr2025/ready-SR-1seq-128rs-s42",
#     "SR-s45": "ckpts_iclr2025/ready-SR-1seq-128rs-s45",
#     "SR-s49": "ckpts_iclr2025/ready-SR-1seq-128rs-s49",
#     "EI-s42": "ckpts_iclr2025/ready-EI-1seq-128rs-s42",
#     "EI-s45": "ckpts_iclr2025/ready-EI-1seq-128rs-s45",
#     "EI-s49": "ckpts_iclr2025/ready-EI-1seq-128rs-s49",
#     "PI-s42": "ckpts_iclr2025/ready-PI-1seq-128rs-s42",
#     "PI-s45": "ckpts_iclr2025/ready-PI-1seq-128rs-s45",
#     "PI-s49": "ckpts_iclr2025/ready-PI-1seq-128rs-s49",
#     "UCB-s42": "ckpts_iclr2025/ready-UCB-1seq-128rs-s42",
#     "UCB-s45": "ckpts_iclr2025/ready-UCB-1seq-128rs-s45",
#     "UCB-s49": "ckpts_iclr2025/ready-UCB-1seq-128rs-s49",
#     "KG-s42": "ckpts_iclr2025/ready-KG-1seq-128rs-s42",
#     "KG-s45": "ckpts_iclr2025/ready-KG-1seq-128rs-s45",
#     "KG-s49": "ckpts_iclr2025/ready-KG-1seq-128rs-s49",
#     "Ours-s42": "ckpts_iclr2025/ready-HES-11-1seq-128rs-s42",
#     "Ours-s45": "ckpts_iclr2025/ready-HES-11-1seq-128rs-s45",
#     "Ours-s49": "ckpts_iclr2025/ready-HES-11-1seq-128rs-s49",
# }

# M1F2
LIST_RESULTS = {
    # "SR-s42": "ckpts/m1f2-SR-1seq-64rs-s42",
    # "SR-s45": "ckpts/m1f2-SR-1seq-64rs-s45",
    # "SR-s49": "ckpts/m1f2-SR-1seq-64rs-s49",
    "EI-s42": "ckpts/m1f2-EI-1seq-64rs-s42",
    "EI-s45": "ckpts/m1f2-EI-1seq-64rs-s45",
    "EI-s49": "ckpts/m1f2-EI-1seq-64rs-s49",
    # "PI-s42": "ckpts/m1f2-PI-1seq-64rs-s42",
    # "PI-s45": "ckpts/m1f2-PI-1seq-64rs-s45",
    # "PI-s49": "ckpts/m1f2-PI-1seq-64rs-s49",
    # "UCB-s42": "ckpts/m1f2-UCB-1seq-64rs-s42",
    # "UCB-s45": "ckpts/m1f2-UCB-1seq-64rs-s45",
    # "UCB-s49": "ckpts/m1f2-UCB-1seq-64rs-s49",
    # "KG-s42": "ckpts/m1f2-KG-1seq-64rs-s42",
    # "KG-s45": "ckpts/m1f2-KG-1seq-64rs-s45",
    # "KG-s49": "ckpts/m1f2-KG-1seq-64rs-s49",
    "Ours-s42": "ckpts/m1f2-HES-11-1seq-64rs-s42",
    "Ours-s45": "ckpts/m1f2-HES-11-1seq-64rs-s45",
    "Ours-s49": "ckpts/m1f2-HES-11-1seq-64rs-s49",
}

LIST_ALGOS = [
    # "SR",
    "EI",
    # "PI",
    # "UCB",
    # "KG",
    "Ours",
]

SEEDS = [42, 49]  # 45,

oracle = joblib.load(f"ckpts/oracle/model.joblib")
MAX_RW = 3
CUT_OFF = 16

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutant_ver", type=str, default="v1")
    parser.add_argument("--fn_ver", type=str, default="v1")
    args = parser.parse_args()

    _, INIT_SEQ, _, _, _ = import_protein_env(args.mutant_ver)

    print("Drawing actual scores...")
    fig, axs = plt.subplots(
        nrows=1, ncols=2, figsize=figsizes.iclr2024(nrows=1, ncols=2)["figure.figsize"]
    )
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    res_dict = {}
    for algo in LIST_ALGOS:
        algo_res = []
        print(algo)
        for seed in SEEDS:
            folder = LIST_RESULTS[algo + "-s" + str(seed)]
            if not os.path.exists(os.path.join(folder, "buffer.pkl")):
                continue
            buffer = pickle.load(open(os.path.join(folder, "buffer.pkl"), "rb"))
            print("Initial:", buffer["x"][0])
            print("Final:", buffer["x"][-1])

            metrics = np.array(buffer["y"])[: CUT_OFF + 2]  # y_X
            # metrics = np.cumsum(metrics, axis=0) # c_regret
            # if metrics.shape[0] != CUT_OFF+2:
            #     continue

            mean = np.mean(metrics, axis=1)
            algo_res.append(mean)

        # min_length = min([len(x) for x in algo_res])
        # algo_res = [x[:min_length] for x in algo_res]
        algo_res = np.stack(algo_res)
        res_dict[algo] = algo_res

    first_res = [v[:, 0].mean() for v in res_dict.values()]
    first_res = np.stack(first_res).mean(axis=0)
    first_mean = first_res.mean()
    first_std = first_res.std()

    for i, (algo, res) in enumerate(res_dict.items()):
        mean = np.mean(res, axis=0)
        std = np.std(res, axis=0)
        mean[0] = first_mean
        std[0] = first_std
        steps = np.arange(0, mean.shape[0])
        if i == 5:
            i = 6
        axs[0].plot(steps, mean, label=algo, color=color_cycle[i])
        axs[0].fill_between(
            steps, mean - std, mean + std, alpha=0.2, color=color_cycle[i]
        )

    # axs[0].tick_params(labelsize=18)
    axs[0].set_xlabel("BO Steps")
    axs[0].set_ylabel("Fluorescence level")

    print("Drawing scores in imagination...")
    # plt.figure(figsize=(10,5))
    for i, algo in enumerate(LIST_ALGOS):
        algo_res = []
        for seed in SEEDS:
            result = LIST_RESULTS[algo + "-s" + str(seed)]
            if not os.path.exists(os.path.join(result, "buffer.pkl")):
                continue

            list_files = os.listdir(os.path.join(result))
            list_files = [x for x in list_files if x.startswith("trajectory")]
            list_files = sorted(list_files)[: CUT_OFF + 1]
            buffer = pickle.load(open(os.path.join(result, "buffer.pkl"), "rb"))

            list_vals = [MAX_RW - buffer["y"][0][0]]
            for traj_file in list_files:
                best_idx, rewards, X_returned = pickle.load(
                    open(os.path.join(result, traj_file), "rb")
                )
                # list_vals.append(rewards[best_idx[0]][0])
                best_emb = get_embedding_from_server(
                    server_url="http://hyperturing1:1338",
                    list_sequences=[X_returned[best_idx[0]][-1][0]],
                )
                best_y = observe_value(
                    oracle,
                    torch.tensor(best_emb),
                    compute_ed(INIT_SEQ, [X_returned[best_idx[0]][-1][0]]),
                    args.fn_ver,
                )[0]
                best_y = MAX_RW - best_y  # regret
                list_vals.append(best_y)

            # list_vals = np.cumsum(np.array(list_vals)) # cum regret
            # if len(list_vals) != CUT_OFF+2:
            #     continue
            algo_res.append(list_vals)
            print(algo, f"{seed}", list_vals)

        # min_length = min([len(x) for x in algo_res])
        # algo_res = [x[:min_length] for x in algo_res]
        # steps = list(range(min_length))

        steps = list(range(CUT_OFF + 1))
        algo_res = np.stack(algo_res)

        # for _ in range(10):
        #     select_idxs = np.random.randint(0, len(SEEDS), SEEDS
        #     mean = algo_res.mean(axis=0)
        mean = algo_res.mean(axis=0)
        std = algo_res.std(axis=0)

        if i == 5:
            i = 6
        axs[1].plot(steps, mean, label=algo, color=color_cycle[i])
        axs[1].fill_between(
            steps, mean - std, mean + std, alpha=0.2, color=color_cycle[i]
        )

    axs[1].set_yscale("log")
    # axs[1].tick_params(labelsize=18)
    axs[1].yaxis.set_major_formatter(ScalarFormatter())
    axs[1].set_xlabel("BO Steps")
    axs[1].set_ylabel("Regret")

    handles, _ = plt.gca().get_legend_handles_labels()
    fig.legend(
        handles,
        LIST_ALGOS,
        loc="outside lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.125),
        # fontsize=20,
    )
    ensure_dir(f"plots/mutant{args.mutant_ver}_fn{args.fn_ver}")
    plt.savefig(f"plots/mutant{args.mutant_ver}_fn{args.fn_ver}/yA_regret.png", dpi=300)
    plt.close()
