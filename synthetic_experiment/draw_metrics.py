import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as mplPath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from tueplots import bundles

plt.rcParams.update(bundles.neurips2024())

algos_name = [
    # "HES-TS-AM-1",
    # "HES-TS-AM-10",
    # "HES-TS-AM-20",
    # "HES-TS-1",
    # "HES-TS-2",
    # "HES-TS-3",
    # "HES-AM-1",
    # "HES-AM-2",
    # "HES-AM-3",
    # "HES-1",
    # "HES-2",
    # "HES-3",
    "MSL-3",
    "SR",
    "EI",
    "PI",
    "UCB",
    "KG",
    "HES-TS-20",
]

algos = [
    # "HES-TS-AM-1",
    # "HES-TS-AM-10",
    # "HES-TS-AM-20",
    # "HES-TS-1",
    # "HES-TS-2",
    # "HES-TS-3",
    # "HES-AM-1",
    # "HES-AM-2",
    # "HES-AM-3",
    # "HES-1",
    # "HES-2",
    # "HES-3",
    "qMSL",
    "qSR",
    "qEI",
    "qPI",
    "qUCB",
    "qKG",
    "HES-TS-20",
]

seeds = [
    2,
    # 3, 5, 7, 11
]

env_names = [
    "Ackley",
    "Alpine",
    "Beale",
    "Branin",
    "Cosine8",
    "EggHolder",
    "Griewank",
    "Hartmann",
    "HolderTable",
    "Levy",
    "Powell",
    "SixHumpCamel",
    "StyblinskiTang",
    "SynGP",
]

env_noises = [
    # 0.0,
    0.01,
    # 0.1,
]

env_discretizeds = [
    False,
    # True
]

cost_functions = ["euclidean", "manhattan", "r-spotlight", "non-markovian"]

cost_function_names = {
    "euclidean": "Euclidean cost",
    "manhattan": "Manhattan cost",
    "r-spotlight": "$r$-spotlight cost",
    "non-markovian": "Non-Markovian cost",
}


def get_env_info(env_name, device):
    if env_name == "Ackley":
        x_dim = 2
        bounds = [-2, 2]
        n_initial_points = 50
        algo_n_iterations = 100

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
            list_means.append(np.mean(mean))

        data_tuple.append([algo, np.mean(list_means), np.std(list_means)])

    data_tuple.sort(key=lambda x: x[1])
    algos = []
    for i, (algo, mean, std) in enumerate(data_tuple):
        if algo == "HES-TS-20":
            algo = "Our"
        elif algo == "MSL-3":
            algo = "MSL"
        algos.append(algo)
        plt.errorbar(i, mean, yerr=std, elinewidth=0.75, fmt="o", ms=5, color="black")

    plt.xticks(ticks=list(range(len(algos))), labels=algos, rotation=90)
    plt.tick_params(axis="both", labelsize=18)

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
            ncol=7,
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

        fig, axs = plt.subplots(
            nrows=len(env_noises),
            ncols=len(cost_functions),
            figsize=(4 * len(cost_functions), 4 * len(env_noises)),
            subplot_kw=dict(projection="radar"),
        )
        # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

        for eni, env_noise in enumerate(env_noises):
            for cfi, cost_fn in enumerate(cost_functions):
                if len(env_noises) == 1:
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
                            list_means.append(0)
                            list_stds.append(0.1)

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

                    ap = ax.plot(list_thetas, list_means)
                    upb = [mean + std for mean, std in zip(list_means, list_stds)]
                    lob = [mean - std for mean, std in zip(list_means, list_stds)]
                    ax.fill_between(
                        list_thetas, upb, lob, alpha=0.25, label="_nolegend_"
                    )
                ax.set_rgrids([-0.6, -0.2, 0.2, 0.6])
                ax.set_title(
                    r"$\sigma =$" + f" {env_noise} - {cost_function_names[cost_fn]}",
                    weight="bold",
                    size="xx-large",
                    position=(0.5, 1.1),
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                ax.set_varlabels(["" for _ in env_names])  # Environments
                # ax.set_varlabels(env_names) # Environments

        # fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
        #          horizontalalignment='center', color='black', weight='bold',
        #          size='large')

        fig.legend(
            algos_name,
            loc="outside lower center",
            ncol=7,
            # bbox_to_anchor=(0.5, -0.075),
            bbox_to_anchor=(0.5, -0.2),
            fontsize=20,
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

    print("Drawing runtime...")
    for env_discretized in env_discretizeds:
        dict_metrics = {}
        for env_name in env_names:
            x_dim, bounds, radius, n_initial_points, algo_n_iterations = get_env_info(
                env_name, device
            )
            for env_noise in env_noises:
                for cost_fn in cost_functions:
                    for aid, algo in enumerate(algos):
                        list_metrics = []
                        for seed in seeds:
                            buffer_file = f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/buffer.pt"
                            if not os.path.exists(buffer_file) and not os.path.exists(
                                f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy"
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

        save_dir = Path("plots/runtime")
        save_dir.mkdir(parents=True, exist_ok=True)
        draw_time(
            "",
            dict_metrics,
            f"plots/runtime{'_discretized' if env_discretized else ''}.png",
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

        dict_metrics = {}
        for aid, algo in enumerate(algos):
            list_metrics = []
            for seed in seeds:
                if os.path.exists(
                    f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy"
                ):
                    metrics = np.load(
                        f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy"
                    )
                else:
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
            "plots/final_cregret.png",
            "plots/final_yA.png",
            "plots/iteration_at_90perc_cregret.png",
        ],
        env_discretized=0,
    )
