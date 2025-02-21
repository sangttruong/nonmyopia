import random

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from tqdm import tqdm
from tueplots import bundles, figsizes

plt.rcParams.update(bundles.iclr2024())


hf_embedding_names = [
    "stair-lab/proteinea_fluorescence-Llama-2-7b-hf-embedding",
    "stair-lab/proteinea_fluorescence-Mistral-7B-v0.1-embedding",
    "stair-lab/proteinea_fluorescence-Meta-Llama-3-8B-embedding",
    "stair-lab/proteinea_fluorescence-gemma-7b-embedding",
    "stair-lab/proteinea_fluorescence-esm2_t36_3B_UR50D-embedding",
    "stair-lab/proteinea_fluorescence-esm2_t33_650M_UR50D-embedding",
    "stair-lab/proteinea_fluorescence-llama-molinst-protein-7b-embedding",
]
r2s_emb = {}
r2s_std_emb = {}
rmse_emb = {}
rmse_std_emb = {}
emb = [
    "Llama-2 7B",
    "Mistral 7B",
    "Llama-3 8B",
    "Gemma 7B",
    "ESM-2 3B",
    "ESM-2 650M",
    "Llama Molinst Protein 7B",
]
total_X = None
for i, hf_embedding_name in enumerate(hf_embedding_names):
    ds = load_dataset(hf_embedding_name)

    X_train = ds["train"].data["inputs_embeds"].to_numpy()
    y_train = ds["train"].data["reward"].to_numpy()
    if total_X is None:
        total_X = X_train.shape[0]

    X_test = ds["test"].data["inputs_embeds"].to_numpy()
    y_test = ds["test"].data["reward"].to_numpy()
    X_test = np.stack(X_test)
    y_test = np.stack(y_test)

    r2s_emb[emb[i]] = []
    r2s_std_emb[emb[i]] = []
    rmse_emb[emb[i]] = []
    rmse_std_emb[emb[i]] = []

    for perc in np.arange(0.1, 1.1, 0.1):
        r2s_local = []
        rmse_local = []
        for s in [2, 3, 5, 7, 11]:
            random.seed(s)
            idxs = random.choices(range(total_X), k=int(perc * total_X))
            X_train_local = np.stack([x for i, x in enumerate(X_train) if i in idxs])
            y_train_local = np.stack([x for i, x in enumerate(y_train) if i in idxs])
            model = BayesianRidge().fit(X_train_local, y_train_local)

            y_hat = model.predict(X_test)
            r2s_local.append(r2_score(y_true=y_test, y_pred=y_hat))
            rmse_local.append(root_mean_squared_error(y_true=y_test, y_pred=y_hat))

        r2s_local = np.array(r2s_local)
        r2s_local_mean = np.mean(r2s_local)
        r2s_local_std = np.std(r2s_local)

        rmse_local = np.array(rmse_local)
        rmse_local_mean = np.mean(rmse_local)
        rmse_local_std = np.std(rmse_local)

        print(f"[{emb[i]}-{perc}]: R2, RMSE are {r2s_local}, {rmse_local}")
        r2s_emb[emb[i]].append(r2s_local_mean)
        r2s_std_emb[emb[i]].append(r2s_local_std)
        rmse_emb[emb[i]].append(rmse_local_mean)
        rmse_std_emb[emb[i]].append(rmse_local_std)
import pickle

with open("oracle_results.pkl", "wb") as f:
    results = (r2s_emb, r2s_std_emb, rmse_emb, rmse_std_emb)
    pickle.dump(results, f)

# total_X = 21446
# with open("oracle_results.pkl", "rb") as f:
#     r2s_emb, r2s_std_emb, rmse_emb, rmse_std_emb = pickle.load(f)

# Draw plots
figsize = figsizes.iclr2024(nrows=1, ncols=1)["figure.figsize"]
plt.figure(figsize=[figsize[0] * 0.8, figsize[1] * 0.5])
sizes = [total_X * p for p in np.arange(0.1, 1.1, 0.1)]
for emb_name in emb:
    plt.plot(sizes, r2s_emb[emb_name], label=emb_name)
    plt.fill_between(
        sizes,
        np.array(r2s_emb[emb_name]) - np.array(r2s_std_emb[emb_name]),
        np.array(r2s_emb[emb_name]) + np.array(r2s_std_emb[emb_name]),
        alpha=0.1,
    )
# plt.ylim(top=1)
plt.ylabel(r"$R^2$")
plt.xlim(left=2048)
plt.xlabel("Number of training data")
plt.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.66))
plt.savefig("plots/R2_convergence.png", dpi=300)
plt.close()

plt.figure(figsize=[figsize[0] * 0.8, figsize[1] * 0.5])
for emb_name in emb:
    plt.plot(sizes, rmse_emb[emb_name], label=emb_name)
    plt.fill_between(
        sizes,
        np.array(rmse_emb[emb_name]) - np.array(rmse_std_emb[emb_name]),
        np.array(rmse_emb[emb_name]) + np.array(rmse_std_emb[emb_name]),
        alpha=0.1,
    )
# plt.ylim(bottom=0)
plt.ylabel(r"$RMSE$")
plt.xlim(left=2048)
plt.xlabel("Number of training data")
plt.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.66))
plt.savefig("plots/RMSE_convergence.png", dpi=300)
plt.close()
