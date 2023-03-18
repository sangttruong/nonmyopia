import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pickle

def uniform_random_sample_domain(domain, n=1):
    """Draws a sample uniformly at random from domain (a list of tuple bounds)."""
    list_of_arr_per_dim = [np.random.uniform(
        dom[0], dom[1], n) for dom in domain]
    list_of_list_per_sample = [list(l)
                               for l in np.array(list_of_arr_per_dim).T]
    return list_of_list_per_sample

default_path_str = "experiments/opt_topk_synthfunc_hes_seed10_n_samples8_n_restarts128_amortized_viTrue_lookahead_steps1_n_layers2_activationrelu_hidden_coeff1_acq_opt_lr0.1_acq_opt_iter200_n_resampling_max1_baselineFalse_05"

def get_init_data(
    path_str=default_path_str, 
    file_str="trial_info.pkl", 
    start_iter=27, 
    n_init_data=10
):
    # Unpickle trial_info Namespace
    with open(path_str + "/" + file_str, "rb") as file:
        trial_info = pickle.load(file)

    init_data = trial_info.config.init_data

    # To initialize directly *before* start_iter
    crop_idx = n_init_data + start_iter - 1
    init_data.x = init_data.x[:crop_idx]
    init_data.y = init_data.y[:crop_idx]

    return init_data
