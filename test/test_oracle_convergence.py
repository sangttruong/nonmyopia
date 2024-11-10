import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from tueplots import bundles

bundles.neurips2024()
plt.rcParams.update(bundles.neurips2024())

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, nargs="+")
parser.add_argument("--models", type=str, nargs="+")
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--output_dir", type=str, default="results")
args = parser.parse_args()

hf_embedding_names = args.datasets
r2s_emb = {}
r2s_std_emb = {}
rmse_emb = {}
rmse_std_emb = {}
models = args.models

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

total_X = None
for i, hf_embedding_name in enumerate(hf_embedding_names):
    ds = load_dataset(hf_embedding_name)

    X_train = ds["train"].data["inputs_embeds"].to_numpy()
    y_train = ds["train"].data["rewards"].to_numpy()
    if total_X is None:
        total_X = X_train.shape[0]

    X_test = ds["validation"].data["inputs_embeds"].to_numpy()
    y_test = ds["validation"].data["rewards"].to_numpy()
    X_test = np.stack(X_test)
    y_test = np.stack(y_test)

    r2s_emb[models[i]] = []
    r2s_std_emb[models[i]] = []
    rmse_emb[models[i]] = []
    rmse_std_emb[models[i]] = []

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

        print(f"[{models[i]}-{perc}]: R2, RMSE are {r2s_local}, {rmse_local}")
        r2s_emb[models[i]].append(r2s_local_mean)
        r2s_std_emb[models[i]].append(r2s_local_std)
        rmse_emb[models[i]].append(rmse_local_mean)
        rmse_std_emb[models[i]].append(rmse_local_std)

with open(os.path.join(args.output_dir, "oracle_results.pkl"), "wb") as f:
    results = (r2s_emb, r2s_std_emb, rmse_emb, rmse_std_emb)
    pickle.dump(results, f)

# with open("oracle_results.pkl", "rb") as f:
#     r2s_emb, r2s_std_emb, rmse_emb, rmse_std_emb = pickle.load(f)

# Draw plots
plt.figure()
sizes = [total_X * p for p in np.arange(0.1, 1.1, 0.1)]
for model_name in models:
    plt.plot(sizes, r2s_emb[model_name], label=model_name)
    plt.fill_between(
        sizes,
        np.array(r2s_emb[model_name]) - np.array(r2s_std_emb[model_name]),
        np.array(r2s_emb[model_name]) + np.array(r2s_std_emb[model_name]),
        alpha=0.1,
    )
plt.ylim(top=1)
plt.ylabel(r"$R^2$")
plt.xlabel("Number of training data")
plt.legend()
plt.savefig(os.path.join(args.output_dir, "R2_convergence.png"), dpi=500)
plt.close()

plt.figure()
for model_name in models:
    plt.plot(sizes, rmse_emb[model_name], label=model_name)
    plt.fill_between(
        sizes,
        np.array(rmse_emb[model_name]) - np.array(rmse_std_emb[model_name]),
        np.array(rmse_emb[model_name]) + np.array(rmse_std_emb[model_name]),
        alpha=0.1,
    )
plt.ylim(bottom=0)
plt.ylabel(r"$RMSE$")
plt.xlabel("Number of training data")
plt.legend()
plt.savefig(os.path.join(args.output_dir, "RMSE_convergence.png"), dpi=500)
plt.close()
