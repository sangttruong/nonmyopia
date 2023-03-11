import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import neatplot


neatplot.set_style()
neatplot.update_rc('axes.grid', False)
neatplot.update_rc('font.size', 16)


label_dict = {
    'hes': '$H_{\ell, \mathcal{A}}$-Entropy Search',
    'kg': 'Knowledge Gradient',
    'us': 'Uncertainty Sampling',
    'rs': 'Random Search',
}


def get_eval_list(exp_str='mvs_mh', algo='hes', seed=1):
    """Return eval_list from trial_file."""

    exp_dir_str = 'experiments/trial_files'
    file_str = 'trial_info_0' + str(seed) + '.pkl'

    file_path = Path(exp_dir_str) / exp_str / algo / file_str

    with open(str(file_path), 'rb') as file_handle:
        trial_info = pickle.load(file_handle)

    eval_list = trial_info.eval_list
    return eval_list


def plot_trial(exp_str, trial_len):
    """Plot a trial."""

    fig, ax = plt.subplots(figsize=(6, 4))

    for algo in ['rs', 'kg', 'us', 'hes']:

        el_list = []
        for seed in [1, 2, 3]:
            eval_list = get_eval_list(exp_str=exp_str, algo=algo, seed=seed)
            eval_list = eval_list[:trial_len]
            if len(eval_list) < trial_len:
                eval_list_buff = [eval_list[-1] for _ in range(trial_len - len(eval_list))]
                eval_list = eval_list + eval_list_buff
            el_list.append(eval_list)

        eval_mean = np.mean(el_list, 0)
        eval_std = np.std(el_list, 0)

        iters = list(range(len(eval_mean)))
        ax.plot(iters, eval_mean, label=label_dict[algo])
        lcb = eval_mean - eval_std / np.sqrt(len(el_list))
        ucb = eval_mean + eval_std / np.sqrt(len(el_list))
        ax.fill_between(iters, lcb, ucb, alpha=0.1)

    return ax



xlabel = 'Iteration'
#ylabel = '$-H_{\ell, \mathcal{A}}[f \mid \mathcal{D}_t]$'
#ylabel = '$-\ell(f, a^*)$'
ylabel = 'Negative Loss'

ax = plot_trial(exp_str='mvs_mh', trial_len=80)
ax.set(ylim=[-3, 0], xlabel=xlabel, ylabel=ylabel)
title = 'Multihills'
ax.set_title(label=title, fontdict={'fontsize': 21})
ax.legend()
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
neatplot.save_figure('mvs_mh', 'pdf')

ax = plot_trial(exp_str='mvs_vacc', trial_len=80)
ax.set(ylim=[-0.33, -0.10], xlabel=xlabel, ylabel=ylabel)
title = 'Vaccination'
ax.set_title(label=title, fontdict={'fontsize': 21})
#ax.legend()
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
neatplot.save_figure('mvs_vacc', 'pdf')

plt.show()
