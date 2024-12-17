import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as mplPath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from tueplots import bundles, figsizes
from utils import ensure_dir

plt.rcParams.update(bundles.iclr2024())


parser = ArgumentParser()
parser.add_argument("--setting", type=str, default="default")
parser.add_argument("--n_initial_points", type=int, default=-1)
parser.add_argument("--n_restarts", type=int, default=64)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--kernel", type=str, default="RBF")
parser.add_argument("--result_dir", type=str, default="./results")
args = parser.parse_args()

line_styles = {
    "SR": (0, (1, 1)),
    "EI": (0, (5, 1)),
    "PI": (0, (3, 5, 1, 5, 1, 5)),
    "UCB": "-.",
    "KG": ":",
    "MSL": "--",
    "Ours": "-",
}

line_markers = {
    "SR": (0, (1, 1)),
    "EI": (0, (5, 1)),
    "PI": (0, (3, 5, 1, 5, 1, 5)),
    "UCB": "-.",
    "KG": ">",
    "MSL": "8",
    "Ours": "",
}

line_colors = {
    "SR": "black",
    "EI": "black",
    "PI": "black",
    "UCB": "black",
    "UCB ($\\beta = 0.1$)": "#d1e5f0",
    "UCB ($\\beta = 0.5$)": "#95c5de",
    "UCB ($\\beta = 1$)": "#4393c3",
    "UCB ($\\beta = 2$)": "#2166ac",
    "KG": "#ee6677",
    "MSL": "#66ccee",
    "MSL-5": "#66ccee",
    "MSL-10": "#66ccee",
    "MSL-15": "#66ccee",
    "MSL-20": "#66ccee",
    "Ours": "#4477aa",
    "Ours-5": "#4477aa",
    "Ours-10": "#4477aa",
    "Ours-15": "#4477aa",
    "Ours-20": "#4477aa",
}

seeds = [
    2,
    3,
    5,
]

if args.setting == "default":
    algos_name = [
        "SR",
        "EI",
        "PI",
        "UCB",
        "KG",
        "MSL",
        "Ours",
    ]

    algos = [
        "qSR",
        "qEI",
        "qPI",
        "qUCB",
        "qKG",
        "HES-TS-20",
        "HES-TS-AM-20",
    ]

    env_names = [
        "Ackley",
        "Ackley4D",
        "Alpine",
        # "Beale",
        # "Branin",
        "Cosine8",
        # "EggHolder",
        # "Griewank",
        "Hartmann",
        "HolderTable",
        "Levy",
        # "Powell",
        # "SixHumpCamel",
        "StyblinskiTang",
        "SynGP",
    ]

    env_noises = [
        # 0.0,
        # 0.01,
        0.05,
    ]

    env_discretizeds = [
        False,
        # True
    ]

    cost_functions = ["euclidean", "manhattan", "r-spotlight", "non-markovian"]

elif args.setting == "lookahead":
    from ablation_configs.lookahead import (
        algos,
        algos_name,
        cost_functions,
        env_discretizeds,
        env_names,
        env_noises,
    )
elif args.setting == "restart":
    from ablation_configs.restart import (
        algos,
        algos_name,
        cost_functions,
        env_discretizeds,
        env_names,
        env_noises,
    )
elif args.setting == "network":
    from ablation_configs.network import (
        algos,
        algos_name,
        cost_functions,
        env_discretizeds,
        env_names,
        env_noises,
    )
elif args.setting == "kernel":
    from ablation_configs.kernel import (
        algos,
        algos_name,
        cost_functions,
        env_discretizeds,
        env_names,
        env_noises,
    )
elif args.setting == "ucb":
    from ablation_configs.ucb import (
        algos,
        algos_name,
        cost_functions,
        env_discretizeds,
        env_names,
        env_noises,
    )
elif args.setting == "init_samples_ackley":
    from ablation_configs.init_samples_ackley import (
        algos,
        algos_name,
        cost_functions,
        env_discretizeds,
        env_names,
        env_noises,
    )
elif args.setting == "init_samples_alpine":
    from ablation_configs.init_samples_alpine import (
        algos,
        algos_name,
        cost_functions,
        env_discretizeds,
        env_names,
        env_noises,
    )
elif args.setting == "init_samples_syngp":
    from ablation_configs.init_samples_syngp import (
        algos,
        algos_name,
        cost_functions,
        env_discretizeds,
        env_names,
        env_noises,
    )
elif args.setting == "nightlight":
    from ablation_configs.nightlight import (
        algos,
        algos_name,
        cost_functions,
        env_discretizeds,
        env_names,
        env_noises,
    )

cost_function_names = {
    "euclidean": "$L_2$ cost",
    "manhattan": "$L_1$ cost",
    "r-spotlight": "Spotlight cost",
    "non-markovian": "Non-Markov cost",
}


def get_env_info(env_name, device):
    if env_name == "Ackley":
        x_dim = 2
        bounds = [-2, 2]
        n_initial_points = 50
        algo_n_iterations = 100

    elif env_name == "Ackley4D":
        x_dim = 4
        bounds = [-2, 2]
        n_initial_points = 100
        algo_n_iterations = 200

    elif env_name == "Alpine":
        x_dim = 2
        bounds = [0, 10]
        n_initial_points = 100
        algo_n_iterations = 150

    elif env_name == "Beale":
        x_dim = 2
        bounds = [-4.5, 4.5]
        n_initial_points = 100
        algo_n_iterations = 150

    elif env_name == "Branin":
        x_dim = 2
        bounds = [[-5, 10], [0, 15]]
        n_initial_points = 20
        algo_n_iterations = 70

    elif env_name == "Cosine8":
        x_dim = 8
        bounds = [-1, 1]
        n_initial_points = 200
        algo_n_iterations = 300

    elif env_name == "EggHolder":
        x_dim = 2
        bounds = [-100, 100]
        n_initial_points = 200
        algo_n_iterations = 250

    elif env_name == "Griewank":
        x_dim = 2
        bounds = [-600, 600]
        n_initial_points = 20
        algo_n_iterations = 70

    elif env_name == "Hartmann":
        x_dim = 6
        bounds = [0, 1]
        n_initial_points = 500
        algo_n_iterations = 600

    elif env_name == "HolderTable":
        x_dim = 2
        bounds = [0, 10]
        n_initial_points = 100
        algo_n_iterations = 150

    elif env_name == "Levy":
        x_dim = 2
        bounds = [-10, 10]
        n_initial_points = 100
        algo_n_iterations = 150

    elif env_name == "Powell":
        x_dim = 4
        bounds = [-4, 5]
        n_initial_points = 100
        algo_n_iterations = 200

    elif env_name == "SixHumpCamel":
        x_dim = 2
        bounds = [[-3, 3], [-2, 2]]
        n_initial_points = 40
        algo_n_iterations = 90

    elif env_name == "StyblinskiTang":
        x_dim = 2
        bounds = [-5, 5]
        n_initial_points = 45
        algo_n_iterations = 95

    elif env_name == "SynGP":
        x_dim = 2
        bounds = [-1, 1]
        n_initial_points = 25
        algo_n_iterations = 75

    elif env_name == "NightLight":
        x_dim = 2
        bounds = [-1, 1]
        n_initial_points = 200
        algo_n_iterations = 250

    else:
        raise NotImplementedError

    # if x_dim == 2 and env_name != "NightLight":
    #     radius = 0.075
    # elif x_dim == 2 and env_name == "NightLight":
    #     radius = 0.1
    if x_dim == 2:
        radius = 0.075
    elif x_dim == 4:
        radius = 0.1
    elif x_dim == 6:
        radius = 0.125
    elif x_dim == 8:
        radius = 0.15
    else:
        raise NotImplementedError

    if x_dim == 2:
        radius = 0.075
    elif x_dim == 4:
        radius = 0.1
    elif x_dim == 6:
        radius = 0.125
    elif x_dim == 8:
        radius = 0.15
    else:
        raise NotImplementedError

    bounds = np.array(bounds)
    if bounds.ndim < 2 or bounds.shape[0] < x_dim:
        bounds = np.tile(bounds, [x_dim, 1])
    bounds = torch.tensor(bounds, dtype=torch.double, device=device)

    return x_dim, bounds, radius, n_initial_points, algo_n_iterations


def draw_time(
    env_name,
    dict_metrics,
    save_file,
):
    plt.figure(figsize=(4, 3))
    data_tuple = []
    for idx, (algo, metrics) in enumerate(dict_metrics.items()):
        # mean = np.mean(metrics)
        # std = np.std(metrics)
        list_means = []
        for _ in range(100):
            mean = np.random.choice(metrics.flatten(), size=100, replace=True)
            mean = np.mean(mean)
            if algo == "HES-TS-20" or algo == "HES-TS-AM-20":
                mean = mean * 3 / 20
            list_means.append(mean)

        data_tuple.append([algo, np.mean(list_means) * 3 / 20, np.std(list_means)])

    data_tuple.sort(key=lambda x: x[1])
    algos = []
    list_colors = []
    for i, (algo, mean, std) in enumerate(data_tuple):
        color = "black"
        if algo == "HES-TS-20":
            algo = "Our$_{T}$"
            color = "red"
        elif algo == "HES-TS-AM-20":
            algo = "Ours"
            color = "red"
        elif algo == "MSL-3":
            algo = "MSL"
        algos.append(algo)
        list_colors.append(color)
        plt.errorbar(i, mean, yerr=std, elinewidth=0.75, fmt="o", ms=5, color=color)

    plt.xticks(ticks=list(range(len(algos))), labels=algos, rotation=-90)
    plt.tick_params(axis="both", labelsize=18)
    for label, color in zip(plt.gca().get_xticklabels(), list_colors):
        label.set_color(color)

    plt.ylabel("Time (s)", fontsize=18)
    plt.title(f"{env_name}", fontsize=20)
    plt.savefig(save_file, dpi=300)
    plt.close()


def draw_metric_v2(
    metric_names,
    all_results,
    save_files,
):
    for mi, metric_name in enumerate(metric_names):
        fig, axs = plt.subplots(
            len(env_noises),
            len(cost_functions),
            figsize=(4 * len(cost_functions), 3 * len(env_noises)),
        )

        for eni, env_noise in enumerate(env_noises):
            for cfi, cost_fn in enumerate(cost_functions):
                for algo in algos_name:
                    list_eids = []
                    list_means = []
                    list_stds = []

                    for eid, env_name in enumerate(env_names):
                        for env_discretized in env_discretizeds:
                            if (
                                env_name,
                                env_noise,
                                env_discretized,
                                cost_fn,
                            ) not in all_results:
                                continue

                            if (
                                algo
                                not in all_results[
                                    (env_name, env_noise, env_discretized, cost_fn)
                                ]
                            ):
                                continue

                            result = all_results[
                                (env_name, env_noise, env_discretized, cost_fn)
                            ][algo]
                            mean = np.mean(
                                result,
                                axis=0,
                            )[mi]
                            std = np.std(
                                result,
                                axis=0,
                            )[mi]
                            list_eids.append(eid)
                            list_means.append(mean)
                            list_stds.append(std)

                    # axs[eni][cfi].scatter(list_eids, list_means, label=algo, marker=".")
                    axs[cfi].errorbar(
                        list_eids,
                        list_means,
                        yerr=list_stds,
                        elinewidth=0.75,
                        fmt="o",
                        ms=5,
                        label=algo,
                        alpha=0.5,
                    )
                axs[cfi].set_xticks([])
                axs[cfi].tick_params(axis="both", labelsize=18)
                axs[cfi].set_title(
                    r"$\sigma =$" + f" {env_noise} - {cost_function_names[cost_fn]}",
                    fontsize=20,
                )

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=8,
            # bbox_to_anchor=(0.5, -0.075),
            bbox_to_anchor=(0.5, -0.2),
            fontsize=20,
        )
        # fig.suptitle(metric_name, fontsize=12)
        plt.savefig(save_files[mi], dpi=300)
        plt.close()


def draw_metric_v3(
    metric_names,
    all_results,
    save_files,
    env_discretized=0,
):
    for mi, metric_name in enumerate(metric_names):
        theta = radar_factory(len(env_names), frame="polygon")
        original_size = figsizes.iclr2024(
            nrows=len(env_noises), ncols=len(cost_functions)
        )["figure.figsize"]

        fig, axs = plt.subplots(
            nrows=len(env_noises),
            ncols=len(cost_functions),
            # figsize=[original_size[0] * 1.25, original_size[1] * 2],
            figsize=[original_size[0] * 0.5, original_size[1]],
            subplot_kw=dict(projection="radar"),
        )
        # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

        for eni, env_noise in enumerate(env_noises):
            for cfi, cost_fn in enumerate(cost_functions):
                if len(env_noises) == 1 and len(cost_functions) == 1:
                    ax = axs
                elif len(env_noises) == 1:
                    ax = axs[cfi]
                else:
                    ax = axs[eni][cfi]

                for algo in algos_name:
                    list_thetas = []
                    list_means = []
                    list_stds = []

                    for eid, env_name in enumerate(env_names):
                        if (
                            env_name,
                            env_noise,
                            env_discretized,
                            cost_fn,
                        ) not in all_results:
                            continue

                        if (
                            algo
                            not in all_results[
                                (env_name, env_noise, env_discretized, cost_fn)
                            ]
                        ):
                            list_thetas.append(theta[eid])
                            list_means.append(-1)
                            list_stds.append(0.0)

                        else:
                            result = all_results[
                                (env_name, env_noise, env_discretized, cost_fn)
                            ][algo]
                            mean = np.mean(
                                result,
                                axis=0,
                            )[mi]
                            std = np.std(
                                result,
                                axis=0,
                            )[mi]
                            list_thetas.append(theta[eid])
                            list_means.append(mean)
                            list_stds.append(std)

                    if "Our" not in algo and args.setting != "ucb":
                        opacity = 0.4
                        # linewidth=2
                    else:
                        opacity = 1.0
                        # linewidth=2

                    ap = ax.plot(
                        list_thetas, list_means, alpha=opacity, color=line_colors[algo]
                    )
                    upb = [mean + std for mean, std in zip(list_means, list_stds)]
                    lob = [mean - std for mean, std in zip(list_means, list_stds)]
                    ax.fill_between(
                        list_thetas,
                        upb,
                        lob,
                        alpha=opacity / 4,
                        label="_nolegend_",
                        color=line_colors[algo],
                    )
                ax.set_rgrids(
                    [-1, -0.4, 0.4, 1], labels=["-1", "", "", "1"], fontsize=5, angle=0
                )
                ax.set_title(
                    f"{cost_function_names[cost_fn]}",
                    # r"$\sigma =$" + f" {env_noise} - {cost_function_names[cost_fn]}",
                    # weight="bold",
                    fontsize=5,
                    position=(0.5, 1.1),
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                ax.set_varlabels(["" for _ in env_names])  # Environments
                # ax.set_varlabels(env_names) # Environments

        fig.legend(
            algos_name,
            loc="outside lower center",
            ncol=7,
            bbox_to_anchor=(0.5, -0.1),
            fontsize=5,
        )

        plt.savefig(save_files[mi], dpi=300)
        plt.close()


def radar_factory(num_vars, frame="circle"):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return mplPath(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=mplPath.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_dir = f"plots/{args.setting}{'_' + args.kernel if args.kernel else ''}/init{args.n_initial_points}_hidden{args.hidden_dim}_rs{args.n_restarts}"
    ensure_dir(plot_dir)

    print("Drawing runtime...")
    for env_discretized in env_discretizeds:
        dict_metrics = {}
        for env_name in env_names:
            x_dim, bounds, radius, n_initial_points, algo_n_iterations = get_env_info(
                env_name, device
            )

            if args.n_initial_points > -1:
                difference = args.n_initial_points - n_initial_points
                n_initial_points = args.n_initial_points
                algo_n_iterations += difference

            for env_noise in env_noises:
                for cost_fn in cost_functions:
                    for aid, algo in enumerate(algos):
                        list_metrics = []
                        for seed in seeds:
                            save_dir = (
                                f"./results_iclr2025/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}{'_' + args.kernel if args.kernel != 'RBF' else ''}/"
                                f"{algo}_{cost_fn}_seed{seed}"  # _init{n_initial_points}_hidden{args.hidden_dim}_rs{args.n_restarts}"
                            )

                            buffer_file = f"{save_dir}/buffer.pt"

                            if not os.path.exists(buffer_file) and not os.path.exists(
                                f"{save_dir}/metrics.npy"
                            ):
                                continue

                            try:
                                buffer = torch.load(buffer_file, map_location=device)
                            except RuntimeWarning:
                                print(f"Ignore {buffer_file}")
                                continue
                            # >>> n_iterations x 1
                            list_metrics.append(
                                buffer["runtime"][n_initial_points:]
                                .cpu()
                                .unsqueeze(-1)
                                .tolist()
                            )

                        if len(list_metrics) == 0:
                            continue
                        # >>> n_seeds x n_iterations x 1

                        list_metrics = np.array(list_metrics).mean(keepdims=True)
                        if np.isnan(np.sum(list_metrics)):
                            continue

                        if algos_name[aid] not in dict_metrics:
                            dict_metrics[algos_name[aid]] = list_metrics
                        else:
                            dict_metrics[algos_name[aid]] = np.concatenate(
                                [dict_metrics[algos_name[aid]], list_metrics], axis=0
                            )

        if len(dict_metrics) == 0:
            continue

        draw_time(
            "",
            dict_metrics,
            f"{plot_dir}/runtime{'_discretized' if env_discretized else ''}.png",
        )

    # Create all triplet (env_name, env_noise, env_discretized)
    datasets = []
    for env_name in env_names:
        for env_noise in env_noises:
            for env_discretized in env_discretizeds:
                for cost_fn in cost_functions:
                    datasets.append((env_name, env_noise, env_discretized, cost_fn))

    all_results = {}
    for dataset in datasets:
        print("Drawing for dataset", dataset)
        env_name, env_noise, env_discretized, cost_fn = dataset
        x_dim, bounds, radius, n_initial_points, algo_n_iterations = get_env_info(
            env_name, device
        )

        if args.n_initial_points > -1:
            difference = args.n_initial_points - n_initial_points
            n_initial_points = args.n_initial_points
            algo_n_iterations += difference

        dict_metrics = {}
        for aid, algo in enumerate(algos):
            list_metrics = []
            for seed in seeds:
                save_dir = (
                    f"./results_iclr2025/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}{'_' + args.kernel if args.kernel != 'RBF' else ''}/"
                    f"{algo}_{cost_fn}_seed{seed}"  # _init{n_initial_points}_hidden{args.hidden_dim}_rs{args.n_restarts}"
                )

                if os.path.exists(f"{save_dir}/metrics.npy"):
                    metrics = np.load(f"{save_dir}/metrics.npy")
                else:
                    print(f"Missing: {save_dir}/metrics.npy")
                    continue

                # >>> n_iterations x 3
                yA = metrics[-1, 1] / 3  # Get the yA and normalised to (-1, 1)
                fr = metrics[-1, -1]  # Get the final cregret
                i90 = next(
                    x
                    for x, val in enumerate(metrics[:, -1])
                    if val > 0.9 * np.max(metrics[:, -1])
                )  # Iteration exceeds 90% max c-regret
                list_metrics.append([fr, yA, i90])

            if len(list_metrics) == 0:
                print(
                    "Missing ",
                    env_name,
                    env_noise,
                    env_discretized,
                    algo,
                    cost_fn,
                    seed,
                )
                continue
            # >>> n_seeds x n_iterations x 3

            dict_metrics[algos_name[aid]] = np.array(list_metrics)

        all_results[(env_name, env_noise, env_discretized, cost_fn)] = dict_metrics

        # >>> algo x n_seeds x n_iterations x 1

    # Compute mean and std of metris
    result_means = {}
    result_stds = {}
    for key, dict_metrics in all_results.items():
        result_means[key] = {}
        result_stds[key] = {}

        for algo, metrics in dict_metrics.items():
            result_means[key][algo] = np.mean(metrics, axis=0)[1]
            result_stds[key][algo] = np.std(metrics, axis=0)[1]

            print(
                key, algo, "\t\t", result_means[key][algo], "+-", result_stds[key][algo]
            )

    mean_df = pd.DataFrame(result_means)
    mean_df = mean_df.round(2).astype(str)
    mean_df.index.name = "param"
    std_df = pd.DataFrame(result_stds)
    std_df = std_df.round(2).astype(str)
    std_df.index.name = "param"

    res_df = (
        pd.concat([mean_df, std_df], axis=0)
        .groupby("param")
        .agg(lambda mu_std: "±".join(mu_std))
    )
    res_df = res_df.transpose()
    res_df.to_csv(f"./results.csv")
    res_df.to_markdown(f"./results.md")
    res_df.to_latex(f"./results.tex")

    save_dir = Path("plots")
    save_dir.mkdir(parents=True, exist_ok=True)
    draw_metric_v3(
        [
            "Final Cummulative Regret",
            "Final Observed Value",
            "Iteration @ 90\% Cummulative Regret",
        ],
        all_results,
        [
            f"{plot_dir}/final_cregret.png",
            f"{plot_dir}/final_yA.png",
            f"{plot_dir}/iteration_at_90perc_cregret.png",
        ],
        env_discretized=0,
    )
