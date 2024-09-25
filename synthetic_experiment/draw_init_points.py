#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Evaluate and plot."""
import pickle
from argparse import ArgumentParser

import matplotlib.cm as cm
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import numpy as np
import torch
from main import Parameters

from tueplots import bundles
from utils import eval_func, make_env, set_seed, str2bool

plt.rcParams.update(bundles.neurips2024())

if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--task", type=str, default="topk")
    parser.add_argument("--env_name", type=str, default="SynGP")
    parser.add_argument("--env_noise", type=float, default=0.01)
    parser.add_argument("--env_discretized", type=str2bool, default=False)
    parser.add_argument("--algo", type=str, default="HES-TS-AM-20")
    parser.add_argument("--cost_fn", type=str, default="r-spotlight")
    parser.add_argument("--plot", type=str2bool, default=False)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--cont", type=str2bool, default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    parms = Parameters(args)

    # Init environment
    env = make_env(
        name=parms.env_name,
        x_dim=parms.x_dim,
        bounds=parms.bounds,
        noise_std=parms.env_noise,
    )
    env = env.to(
        dtype=parms.torch_dtype,
        device=parms.device,
    )
    initX, initA = pickle.load(
        open(
            f"results/{parms.env_name}_{parms.env_noise}{'_discretized' if parms.env_discretized else ''}/{args.algo}_{args.cost_fn}_seed{parms.seed}/init_points.pkl",
            "rb",
        )
    )

    X_domain, Y_domain = np.array([0, 1]), np.array([0, 1])
    X, Y = np.linspace(*X_domain, 100), np.linspace(*Y_domain, 100)
    X, Y = np.meshgrid(X, Y)
    XY = torch.tensor(np.array([X, Y]))  # >> 2 x 100 x 100
    Z = env(XY.to(parms.device, parms.torch_dtype).reshape(2, -1).T).reshape(X.shape)
    Z = Z.cpu().detach()

    norm = mcolors.Normalize(vmin=0, vmax=parms.algo_lookahead_steps)
    cmap = cm.viridis  # Choose a colormap

    initX = [x.cpu().detach() for x in initX]
    initA = initA.cpu().detach()

    fig, axs = plt.subplots(
        parms.n_restarts,
        1,
        figsize=(4, 4 * parms.n_restarts),
    )
    for ridx in range(parms.n_restarts):
        axs[ridx].contourf(X, Y, Z, levels=30, cmap="bwr", alpha=0.7)
        for i in range(parms.algo_lookahead_steps):
            color = cmap(norm(i))
            if i > 0:
                for ix in range(initX[i].shape[0]):
                    tail = initX[i - 1][..., ridx, :, :].reshape(-1, parms.x_dim)
                    head = initX[i][ix, ..., ridx, :, :].reshape(-1, parms.x_dim)
                    for tp, hp in zip(tail, head):
                        axs[ridx].plot([tp[0], hp[0]], [tp[1], hp[1]], color=color)

            axs[ridx].scatter(
                initX[i][..., ridx, :, 0], initX[i][..., ridx, :, 1], color=color
            )

        axs[ridx].scatter(
            initA[..., ridx, :, 0], initA[..., ridx, :, 1], color="orange"
        )

    plt.savefig(
        f"results/{parms.env_name}_{parms.env_noise}{'_discretized' if parms.env_discretized else ''}/{args.algo}_{args.cost_fn}_seed{parms.seed}/init_points.png",
        dpi=200,
    )
    plt.close()
