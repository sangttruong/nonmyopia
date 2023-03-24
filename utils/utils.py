import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pickle
from argparse import Namespace


def generate_initial_data(env, config):
    data_x = torch.tensor(np.array(
            [np.random.uniform(dom[0], dom[1], config.n_initial_points) for dom in env.domain]
        ).T, device=config.device, dtype=config.torch_dtype
    )
    
    data_y = env.func(data_x)  # n x 1
    if config.func_is_noisy:
        data_y = data_y + config.func_noise * torch.randn_like(
            data_y, config.device, config.torch_dtype
        )
    data = Namespace(x=data_x, y=data_y)
    return data
        
def set_seed(seed):
    """
    Set random seed at a given iteration, using seed and iteration (both positive
    integers) as inputs.
    """
    # First set initial random seed
    torch.manual_seed(seed=seed)
    np.random.seed(seed)

    # Then multiply iteration with a random integer and set as new seed
    torch.manual_seed(seed=seed)
    np.random.seed(seed)


def get_init_data(
    path_str, 
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
