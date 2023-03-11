from argparse import Namespace, ArgumentParser
import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import torch
from botorch.acquisition import PosteriorMean
from src.run import run_hes_trial, uniform_random_sample_domain
from multihills import Multihills, multihills_bounds
import neatplot
import pickle

neatplot.set_style()
neatplot.update_rc('axes.grid', False)
neatplot.update_rc('font.size', 16)
neatplot.update_rc('text.usetex', True)

# Initialize Multihills
centers = [[0.2, 0.15], [0.2, 0.75], [0.75, 0.5]]
widths = [0.2, 0.2, 0.2]
weights = [0.4, 0.8, 0.9]
multihills = Multihills(centers, widths, weights)
threshold = 2.1

plot_data_files = ['levelset_multihills_rs_seed10',
                   'levelset_multihills_us_seed10',
                   'levelset_multihills_kg_seed10',
                   'levelset_multihills_hes_seed30',
                   'levelset_multihills_gpclassifier_seed20']
method_name = ['Random Search', 'Uncertainty Sampling', 'Knowledge Gradient', r'$H_{\ell, \mathcal{A}}$-Entropy Search', 'Probability of Misclassification']

plot_iteration_num = 40

@np.vectorize
def func_vec(x, y):
    """Vectorized function for contour plot. Return f on input = (x, y)."""
    return multihills((x, y))


def plot_function_contour(ax, threshold=None):
    gridwidth = 0.01
    xpts = np.arange(multihills_bounds[0], multihills_bounds[1], gridwidth)
    ypts = np.arange(multihills_bounds[0], multihills_bounds[1], gridwidth)
    X, Y = np.meshgrid(xpts, ypts)
    Z = func_vec(X, Y)
    ax.contour(X, Y, Z, 50, cmap=cm.Greens_r, zorder=0)
    # print(Z.max(), Z.min(), Z.mean())
    if threshold:
        ax.contour(X, Y, Z, levels=[threshold], colors='red', linewidths=3, linestyles='dashed')


def plot_data(ax, data):
    data_x = copy.deepcopy(data.x.detach()).numpy()
    for xi in data_x:
        ax.plot(xi[0], xi[1], 'o', color='black', markersize=6)


for i, file_name in enumerate(plot_data_files):
    trial_info = pickle.load(open(f'./experiments/{file_name}/trial_info.pkl', 'rb'))

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_function_contour(ax=ax, threshold=threshold)
    data_x = trial_info.data.x
    for xi in data_x[:plot_iteration_num]:
        ax.plot(xi[0], xi[1], 'o', color='black', markersize=6)
    ax.set(xlabel='$x_1$', ylabel='$x_2$', xlim=multihills_bounds, ylim=multihills_bounds)
    ax.set_title(label=method_name[i], fontdict={'fontsize': 25})

    # Save plot and close
    neatplot.save_figure(f'{method_name[i]}_levelset_multihills', 'pdf', '/Users/lantaoyu/Desktop/ICLR22')
    plt.close()

plot_data_files = [['levelset_multihills_rs_seed10', 'levelset_multihills_rs_seed20', 'levelset_multihills_rs_seed30'],
                   ['levelset_multihills_us_seed10', 'levelset_multihills_us_seed20', 'levelset_multihills_us_seed30'],
                   ['levelset_multihills_kg_seed10', 'levelset_multihills_kg_seed20', 'levelset_multihills_kg_seed30'],
                   ['levelset_multihills_hes_seed10', 'levelset_multihills_hes_seed20', 'levelset_multihills_hes_seed30'],
                   ['levelset_multihills_gpclassifier_seed20', 'levelset_multihills_gpclassifier_seed40', 'levelset_multihills_gpclassifier_seed70']]

plt.figure(figsize=(6, 4))
for i, data_files in enumerate(plot_data_files):
    all_eval_list = []
    for data_file in data_files:
        trial_info = pickle.load(open(f'./experiments/{data_file}/trial_info.pkl', 'rb'))
        eval_list = trial_info.eval_list
        all_eval_list.append(eval_list[:plot_iteration_num])
    all_eval_list = np.array(all_eval_list)
    all_eval_list_mean = np.mean(all_eval_list, axis=0)
    all_eval_list_std = np.std(all_eval_list, axis=0)
    all_eval_list_mean = all_eval_list_mean
    all_eval_list_std = all_eval_list_std
    plt.plot(all_eval_list_mean, label=method_name[i])
    plt.fill_between(range(plot_iteration_num), all_eval_list_mean - all_eval_list_std, all_eval_list_mean + all_eval_list_std,
                     alpha=0.1)
plt.xlim(0, plot_iteration_num)
# plt.legend(prop={'size': 12})
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Multihills', fontdict={'fontsize': 21})
neatplot.save_figure('levelset_acc_multihills', 'pdf', '/Users/lantaoyu/Desktop/ICLR22')
