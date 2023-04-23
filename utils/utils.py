import pickle
import random
import numpy as np
import torch


def sang_sampler(num_samples=5):
    def sampling(posterior):
        assert num_samples >= 1
        assert num_samples % 2 == 1

        sample = []
        mean = posterior.mean
        std = torch.sqrt(posterior.variance)
        std_coeff = np.linspace(0, 2, num_samples // 2 + 1)
        for s in std_coeff:
            if s == 0:
                sample.append(mean)
            else:
                sample.append(mean + s * std)
                sample.append(mean - s * std)

        out = torch.stack(sample, dim=0)
        return out

    return sampling


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


def get_init_data(path_str, file_str="trial_info.pkl", start_iter=27, n_init_data=10):
    # Unpickle trial_info Namespace
    with open(path_str + "/" + file_str, "rb") as file:
        trial_info = pickle.load(file)

    init_data = trial_info.config.init_data

    # To initialize directly *before* start_iter
    crop_idx = n_init_data + start_iter - 1
    init_data.x = init_data.x[:crop_idx]
    init_data.y = init_data.y[:crop_idx]

    return init_data
