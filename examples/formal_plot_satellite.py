import numpy as np
from matplotlib import cm, pyplot as plt
import pickle
import neatplot

neatplot.set_style()
neatplot.update_rc('axes.grid', False)
neatplot.update_rc('font.size', 16)
neatplot.update_rc('text.usetex', True)

country = "us"  # "us" or "africa"
district = "pennsylvania"  # pennsylvania
sampling_method = "NL"  # nightlight
npy_file = open(f"./satellite_exps/figures/{country}_{district}_{sampling_method}.npy", "rb")
img = np.load(npy_file)
y_lim = img.shape[0] - 1
x_lim = img.shape[1] - 1
target_max = float(img.max())
target_min = float(img.min())


def transform_back_to_original_scale(X):
    X[:, 0] = X[:, 0] * x_lim
    X[:, 1] = (1 - X[:, 1]) * y_lim
    return X


plot_data_files = ['levelset_us-pennsylvania-NL_rs_seed10',
                   'levelset_us-pennsylvania-NL_us_seed10',
                   'levelset_us-pennsylvania-NL_kg_seed20',
                   'levelset_us-pennsylvania-NL_hes_seed10_04']
method_name = ['Random Search', 'Uncertainty Sampling', 'Knowledge Gradient', r'$H_{\ell, \mathcal{A}}$-Entropy Search', 'Probability of Misclassification']

for i, data_file in enumerate(plot_data_files):
    plot_data_path = f'./experiments/{data_file}/plot_data.pkl'
    XYZ, data_x, support_points, x_actions = pickle.load(open(plot_data_path, 'rb'))
    data_x = transform_back_to_original_scale(data_x)
    for xi in data_x[:80]:
        plt.plot(xi[0], xi[1], 'o', color='red', markersize=3)
    plt.imshow(img)

    # plot contour
    X, Y, Z = XYZ
    X = X * x_lim
    Y = (1 - Y) * y_lim
    Z = Z * (target_max - target_min) + target_min
    plt.contour(X, Y, Z, levels=[0.5 * (target_max - target_min) + target_min],
                colors='white', linewidths=3, linestyles='dashed')
    plt.tight_layout()
    plt.axis('off')
    plt.title(method_name[i], fontdict={'fontsize': 12.5})

    # neatplot.save_figure(f'{method_name[i]}_levelset_satellite', 'pdf', f'./experiments/{data_file}')
    neatplot.save_figure(f'{method_name[i]}_levelset_satellite', 'pdf', '/Users/lantaoyu/Desktop/ICLR22')
    plt.close()

plot_data_files = [['levelset_us-pennsylvania-NL_rs_seed10', 'levelset_us-pennsylvania-NL_rs_seed20', 'levelset_us-pennsylvania-NL_rs_seed30'],
                   ['levelset_us-pennsylvania-NL_us_seed10', 'levelset_us-pennsylvania-NL_us_seed20', 'levelset_us-pennsylvania-NL_us_seed30'],
                   ['levelset_us-pennsylvania-NL_kg_seed20', 'levelset_us-pennsylvania-NL_kg_seed15', 'levelset_us-pennsylvania-NL_kg_seed25'],
                   ['levelset_us-pennsylvania-NL_hes_seed10_04', 'levelset_us-pennsylvania-NL_hes_seed20_00', 'levelset_us-pennsylvania-NL_hes_seed30'],
                   ['levelset_us-pennsylvania-NL_gpclassifier_seed10', 'levelset_us-pennsylvania-NL_gpclassifier_seed15', 'levelset_us-pennsylvania-NL_gpclassifier_seed20']]

plt.figure(figsize=(6, 4))
for i, data_files in enumerate(plot_data_files):
    all_eval_list = []
    for data_file in data_files:
        plot_data_path = f'./experiments/{data_file}/eval_list.pkl'
        eval_list = pickle.load(open(plot_data_path, 'rb'))
        all_eval_list.append(eval_list)
    all_eval_list = np.array(all_eval_list)
    all_eval_list_mean = np.mean(all_eval_list, axis=0)
    all_eval_list_std = np.std(all_eval_list, axis=0)
    all_eval_list_mean = all_eval_list_mean[:100]
    all_eval_list_std = all_eval_list_std[:100]
    plt.plot(all_eval_list_mean, label=method_name[i])
    plt.fill_between(range(len(all_eval_list_mean)), all_eval_list_mean - all_eval_list_std, all_eval_list_mean + all_eval_list_std, alpha=0.1)
plt.xlim(0, 100)
plt.ylim(0.6, 0.98)
plt.legend(prop={'size': 15})
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Pennsylvania Night Light', fontdict={'fontsize': 21})
# neatplot.save_figure('levelset_acc', 'pdf', './experiments')
neatplot.save_figure('levelset_acc_satellite', 'pdf', '/Users/lantaoyu/Desktop/ICLR22')
