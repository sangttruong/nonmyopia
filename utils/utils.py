import numpy as np
import torch
import random
from torch import nn
import torch.nn.functional as F
import pickle
from argparse import Namespace
from tqdm import tqdm
from utils.plot import plot_topk

def l2_cost_function(previous_X, current_X, parms):
    
    pass

def splotlight_cost_function(previous_X, current_X, parms):
    # return 0
    diff = torch.sqrt(torch.pow(current_X - previous_X, 2).sum(-1))
    nb_idx = diff < 0.1
    diff = diff * (1 - nb_idx.float()) * 100
    return diff

def stochastic_cost_function(previous_X, current_X, parms):
    
    pass
    
def sang_sampler(num_samples=5):
    def sampling(posterior):
        assert num_samples >= 1
        assert num_samples % 2 == 1

        sample = []
        mean = posterior.mean
        std = torch.sqrt(posterior.variance)
        std_coeff = np.linspace(0, 2, num_samples//2 + 1)
        for s in std_coeff:
            if s == 0: sample.append(mean)
            else:
                sample.append(mean + s*std)
                sample.append(mean - s*std)
        
        out = torch.stack(sample, dim=0)
        return out
    return sampling

def generate_initial_data(env, config):
    data_x = torch.tensor(np.array(
            [np.random.uniform(dom[0], dom[1], config.n_initial_points) for dom in env.domain]
        ).T, dtype=config.torch_dtype
    )
    
    data_y = env.func(data_x)  # n x 1
    if config.func_is_noisy:
        data_y = data_y + config.func_noise * torch.randn_like(
            data_y, config.torch_dtype
        )
    data = Namespace(x=data_x, y=data_y)
    return data
        
def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


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


def eval_topk(config, env, actor, buffer, iteration):
    """Return evaluation metric."""
    eval_metric, optimal_actions = actor.get_topK_actions(
        (buffer.x[-config.n_restarts:], buffer.y[-config.n_restarts:])
    )
    eval_metric = eval_metric.cpu()
    optimal_actions = optimal_actions.cpu()
    
    # Plot optimal_action in special eval plot here
    plot_topk(config=config,
              env=env,
              buffer=buffer,
              iteration=iteration,
              next_x=buffer.x[-1],
              previous_x=buffer.x[-2],
              actions=optimal_actions,
              eval=True)

    # Return eval_metric and eval_data (or None)
    return eval_metric.numpy().tolist(), optimal_actions.numpy().tolist()