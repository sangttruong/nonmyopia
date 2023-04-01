import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pickle
from argparse import Namespace
from tqdm import tqdm
# from botorch.samplers.base import MCSampler
# from torch import Tensor
# from botorch.posteriors import Posterior
from utils.plot import plot_topk

def sang_sampler(num_samples=5):
    def sampling(posterior):
        assert num_samples >= 1
        assert num_samples % 2 == 1

        sample = []
        mean = posterior.mean
        std = torch.sqrt(posterior.variance)
        std_coeff = np.linspace(0, 1, num_samples//2 + 1)
        for s in std_coeff:
            if s == 0: sample.append(mean)
            else:
                sample.append(mean + s*std)
                sample.append(mean - s*std)
        
        out = torch.stack(sample, dim=0)
        return out
    return sampling
# class NormalMCSampler(MCSampler):
#     r"""Base class for samplers producing (possibly QMC) N(0,1) samples.

#     Subclasses must implement the `_construct_base_samples` method.
#     """

#     def forward(self, posterior: Posterior) -> Tensor:
#         r"""Draws MC samples from the posterior.

#         Args:
#             posterior: The posterior to sample from.

#         Returns:
#             The samples drawn from the posterior.
#         """
#         self._construct_base_samples(posterior=posterior)
#         samples = posterior.rsample_from_base_samples(
#             sample_shape=self.sample_shape,
#             base_samples=self.base_samples.expand(
#                 self._get_extended_base_sample_shape(posterior=posterior)
#             ),
#         )
#         return samples

#     def _construct_base_samples(self, posterior: Posterior) -> None:
#         r"""Generate base samples (if necessary).

#         This function will generate a new set of base samples and register the
#         `base_samples` buffer if one of the following is true:

#         - the MCSampler has no `base_samples` attribute.
#         - the output of `_get_collapsed_shape` does not agree with the shape of
#             `self.base_samples`.

#         Args:
#             posterior: The Posterior for which to generate base samples.
#         """
#         pass  # pragma: no cover

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