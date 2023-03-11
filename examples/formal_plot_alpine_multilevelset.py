import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import torch
from botorch.acquisition import PosteriorMean
from src.run import run_hes_trial, uniform_random_sample_domain
from alpine import AlpineN1, plot_alpine_2d
import neatplot
from argparse import Namespace, ArgumentParser
import pickle
import copy

neatplot.set_style()
neatplot.update_rc('axes.grid', False)
neatplot.update_rc('font.size', 16)
neatplot.update_rc('text.usetex', True)

plot_data_files = ['multilevelset_alpine_rs_seed10',
                   'multilevelset_alpine_us_seed10',
                   'multilevelset_alpine_kg_seed10',
                   'multilevelset_alpine_hes_seed10']
method_name = ['Random Search', 'Uncertainty Sampling', 'Knowledge Gradient', r'$H_{\ell, \mathcal{A}}$-Entropy Search']
alpine = AlpineN1(d=2, x_scale=0.5, y_scale=0.05)
alpine_bounds = alpine.get_bounds()


def plot_data(ax, data):
    data_x = copy.deepcopy(data.x.detach()).numpy()
    for xi in data_x:
        xi = alpine.transform_to_domain(xi)
        ax.plot(xi[0], xi[1], 'o', color='black', markersize=6)


def plot_next_query(ax, next_x):
    next_x = copy.deepcopy(next_x.detach()).numpy().reshape(-1)
    next_x = alpine.transform_to_domain(next_x)
    ax.plot(next_x[0], next_x[1], 'o', color='deeppink', markersize=7)


for i, file_name in enumerate(plot_data_files):
    trial_info = pickle.load(open(f'./experiments/{file_name}/trial_info.pkl', 'rb'))

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_alpine_2d(ax=ax, threshold=[0.35 / alpine.y_scale, 0.6 / alpine.y_scale])
    data_x = trial_info.data.x
    for xi in data_x:
        xi = alpine.transform_to_domain(xi)
        ax.plot(xi[0], xi[1], 'o', color='black', markersize=6)
    bounds_plot = alpine.get_bounds(original=True)
    ax.set(xlabel='$x_1$', ylabel='$x_2$', xlim=bounds_plot, ylim=bounds_plot)
    ax.set_title(label=method_name[i], fontdict={'fontsize': 25})

    # Save plot and close
    neatplot.save_figure(f'{method_name[i]}_multilevelset_alpine', 'pdf', '/Users/lantaoyu/Desktop/ICML2022')
    plt.close()

plot_data_files = [['multilevelset_alpine_rs_seed10', 'multilevelset_alpine_rs_seed20', 'multilevelset_alpine_rs_seed30'],
                   ['multilevelset_alpine_us_seed10', 'multilevelset_alpine_us_seed20', 'multilevelset_alpine_us_seed30'],
                   ['multilevelset_alpine_kg_seed10', 'multilevelset_alpine_kg_seed20', 'multilevelset_alpine_kg_seed30'],
                   ['multilevelset_alpine_hes_seed10', 'multilevelset_alpine_hes_seed20', 'multilevelset_alpine_hes_seed30']]

plt.figure(figsize=(6, 4))
for i, data_files in enumerate(plot_data_files):
    all_eval_list = []
    for data_file in data_files:
        trial_info = pickle.load(open(f'./experiments/{data_file}/trial_info.pkl', 'rb'))
        eval_list = trial_info.eval_list
        all_eval_list.append(eval_list)
    all_eval_list = np.array(all_eval_list)
    all_eval_list_mean = np.mean(all_eval_list, axis=0)
    all_eval_list_std = np.std(all_eval_list, axis=0)
    all_eval_list_mean = all_eval_list_mean[:100]
    all_eval_list_std = all_eval_list_std[:100]
    plt.plot(all_eval_list_mean, label=method_name[i])
    plt.fill_between(range(100), all_eval_list_mean - all_eval_list_std, all_eval_list_mean + all_eval_list_std,
                     alpha=0.1)
plt.xlim(0, 100)
plt.legend(prop={'size': 15})
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Alpine-2', fontdict={'fontsize': 21})
neatplot.save_figure('multilevelset_acc_alpine', 'pdf', '/Users/lantaoyu/Desktop/ICML2022')
