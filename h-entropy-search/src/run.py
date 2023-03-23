import time
import copy
from argparse import Namespace
from pathlib import Path
import pickle
import numpy as np
import torch
import os
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_model
from botorch.sampling.normal import SobolQMCNormalSampler
from .hentropy import qHEntropySearch
from .mykg import MyqKnowledgeGradient, initialize_action_tensor_kg
from sklearn.gaussian_process import GaussianProcessClassifier

# Set torch device settings
# torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_device = torch.device("cpu")
torch_dtype = torch.double
torch.set_num_threads(os.cpu_count())


def uniform_random_sample_domain(domain, n, device=torch_device, dtype=torch_dtype):
    """Draws a sample uniformly at random from domain (a list of tuple bounds)."""
    list_of_arr_per_dim = [np.random.uniform(dom[0], dom[1], n) for dom in domain]
    return torch.tensor(np.array(list_of_arr_per_dim).T, device=device, dtype=dtype)


def generate_initial_data(func, config):
    ngen = config.num_initial_points
    domain = [config.bounds_design] * config.num_dim_design
    data_x = uniform_random_sample_domain(domain, ngen)  # n x dim
    data_y = func(data_x)  # n x 1
    if config.func_is_noisy:
        data_y = data_y + config.func_noise * torch.randn_like(data_y, torch_device, torch_dtype)
    data = Namespace(x=data_x, y=data_y)
    return data


def initialize_model(data, state_dict=None):
    model = SingleTaskGP(data.x, data.y).to(data.x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return mll, model


def print_iteration(iteration, next_x, next_y, action_samples, time_iter):
    """Print information at each iteration of HES."""
    print(f'next_x = {next_x.detach().numpy().tolist()}')
    print(f'next_y = {next_y.detach().numpy().tolist()}')
    #print(f'action_samples = {action_samples.detach().numpy().tolist()}')
    print(f'time_iter = {time_iter}.')


def optimize_rs(config):
    """Optimize random search (rs) acquisition function, return next_x."""
    domain = [config.bounds_design] * config.num_dim_design
    data_x = uniform_random_sample_domain(domain, 1)
    next_x = data_x[0].reshape(1, -1)
    return next_x


def optimize_us(model, config):
    """Optimize uncertainty sampling (us) acquisition function, return next_x."""
    domain = [config.bounds_design] * config.num_dim_design
    n_acq_opt_samp = 500
    data_x = uniform_random_sample_domain(domain, n_acq_opt_samp)
    acq_values = model(data_x).variance
    best = torch.argmax(acq_values.view(-1), dim=0)
    next_x = data_x[best].reshape(1, -1)
    return next_x


def optimize_kg(X_design, X_action, model, sampler, config, iteration):
    """Optimize knowledge gradient (kg) acquisition function, return next_x."""
    if not config.num_dim_design == config.num_dim_action:
        X_action = initialize_action_tensor_kg(config)

    qkg = MyqKnowledgeGradient(model, config, sampler)

    optimizer = torch.optim.Adam([X_design, X_action], lr=config.acq_opt_lr)
    for i in range(config.acq_opt_iter):
        optimizer.zero_grad()
        losses = -qkg(X_design, X_action)
        loss = losses.sum()
        loss.backward(retain_graph=True)
        optimizer.step()
        X_design.data.clamp_(config.bounds_design[0], config.bounds_design[1])
        X_action.data.clamp_(config.bounds_action[0], config.bounds_action[1])
        if (i+1) % (config.acq_opt_iter//5) == 0 or i == config.acq_opt_iter-1:
            print(iteration, i+1, loss.item())
    acq_values = qkg(X_design, X_action)
    best = torch.argmax(acq_values.view(-1), dim=0)
    next_x = X_design[best]
    return next_x


def optimize_hes(X_design, X_action, qhes, config, iteration):
    """Optimize hes acquisition function, return acq_vals."""
    optimizer = torch.optim.Adam([X_design, X_action], lr=config.acq_opt_lr)
    for i in range(config.acq_opt_iter):
        optimizer.zero_grad()
        losses = -qhes(X_design, X_action)
        loss = losses.sum()
        loss.backward(retain_graph=True)
        optimizer.step()
        X_design.data.clamp_(config.bounds_design[0], config.bounds_design[1])
        X_action.data.clamp_(config.bounds_action[0], config.bounds_action[1])

        if (i+1) % (config.acq_opt_iter//5) == 0 or i == config.acq_opt_iter-1:
            print(iteration, i+1, loss.item())

    acq_values = qhes(X_design, X_action)
    best = torch.argmax(acq_values.view(-1), dim=0)
    next_x = X_design[best]
    return next_x


def run_hes_trial(
    func,
    config,
    plot_function=None,
    eval_function=None,
    final_eval_function=None,
):
    """Run a single trial of H-Entropy Search using inputs as configuration."""

    # Create save directory
    make_save_dir(config)

    # Set random seeds
    torch.manual_seed(seed=config.seed)
    np.random.seed(config.seed)

    # Generate initial observations and initialize model
    with torch.no_grad():
        data = generate_initial_data(func, config)
    mll_hes, model_hes = initialize_model(data)

    # Logging
    eval_list = []
    eval_data_list = []

    # Run BO loop
    for iteration in range(1, config.num_iteration + 1):

        print('---' * 5 + f' Iteration {iteration} ' + '---' * 5)
        time_start = time.time()

        # Fit the model
        fit_gpytorch_model(mll_hes)

        # Define qhes object
        qmc_sampler = SobolQMCNormalSampler(sample_shape=config.num_outer_mc)
        qhes = qHEntropySearch(model_hes, config, sampler=qmc_sampler)

        # Initialize X_design, X_action
        X_design, X_action = qhes.initialize_design_action_tensors(data)

        # Optimize acquisition function
        if config.algo == 'hes':
            next_x = optimize_hes(X_design, X_action, qhes, config, iteration)
        elif config.algo == 'rs':
            next_x = optimize_rs(config)
        elif config.algo == 'us':
            next_x = optimize_us(model_hes, config)
        elif config.algo == 'kg' or config.algo == 'kgtopk':
            next_x = optimize_kg(X_design, X_action, model_hes, qmc_sampler, config, iteration)

        # End timer
        time_iter = time.time() - time_start

        # For evaluation/plotting, do following for every acq function
        acq_values = qhes(X_design, X_action)
        best = torch.argmax(acq_values.view(-1), dim=0)
        action_samples = X_action[best]

        # Query black box function, observe next_y
        with torch.no_grad():
            next_y = func(next_x)

        # Print
        print_iteration(iteration, next_x, next_y, action_samples, time_iter)

        # Plot
        if plot_function:
            with torch.no_grad():
                plot_function(config, next_x, data, action_samples, iteration)

        # Evaluate
        if eval_function:
            eval_metric, eval_data = eval_function(qhes, config, data, next_x, iteration)
            eval_list.append(eval_metric)
            eval_data_list.append(eval_data)
            print(f'--\nEval metric: {eval_metric}')

        # Update training points
        data.x = torch.cat([data.x, next_x.detach()])
        data.y = torch.cat([data.y, next_y.detach()])

        # Re-initialize model for next iteration, use state_dict to speed up fitting
        mll_hes, model_hes = initialize_model(data, model_hes.state_dict())

        # Pickle trial info at each iteration (overwriting file from previous iteration)
        pickle_trial_info(config, data, eval_list, eval_data_list)

    # Optional final evaluation
    if final_eval_function:
        with torch.no_grad():
            final_eval_function(eval_list, config)


def pickle_trial_info(config, data, eval_list, eval_data_list):
    """Save trial info as a pickle in directory specified by config."""
    # Build trial info Namespace
    data = Namespace(x=data.x.detach().numpy(), y=data.y.detach().numpy())
    trial_info = Namespace(
        config=config, data=data, eval_list=eval_list, eval_data_list=eval_data_list
    )

    # Pickle trial info
    dir_path = Path(str(config.save_dir))
    file_path = dir_path / 'trial_info.pkl'
    with open(str(file_path), 'wb') as file_handle:
        pickle.dump(trial_info, file_handle)


def make_save_dir(config):
    """Create save directory safely (without overwriting directories), using config."""
    init_dir_path = Path(config.save_dir)
    dir_path = Path(str(init_dir_path))

    for i in range(50):
        try:
            dir_path.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            dir_path = Path(str(init_dir_path) + '_' + str(i).zfill(2))

    config.save_dir = str(dir_path)
    print(f'Created save_dir: {config.save_dir}')



def run_gpclassifier_trial(
    func,
    config,
    plot_function=None,
    eval_function=None,
    final_eval_function=None,
):
    """Run a single trial of GP classifier using inputs as configuration."""

    assert config.algo == 'gpclassifier' and (config.app == 'levelset' or config.app == 'multilevelset')

    # Create save directory
    make_save_dir(config)

    # Set random seeds
    torch.manual_seed(seed=config.seed)
    np.random.seed(config.seed)

    # Generate initial observations and initialize model
    with torch.no_grad():
        data = generate_initial_data(func, config)
    mll_hes, model_hes = initialize_model(data)

    def assign_class(y):
        if config.app == 'levelset':
            return 1 if y > config.levelset_threshold else 0
        else:
            label = 0
            for threshold in config.levelset_thresholds:
                if y < threshold:
                    return label
                label += 1
            return label

    # Logging
    eval_list = []
    eval_data_list = []

    # Run BO loop
    for iteration in range(1, config.num_iteration + 1):

        print('---' * 5 + f' Iteration {iteration} ' + '---' * 5)
        time_start = time.time()

        # Fit the model
        fit_gpytorch_model(mll_hes)

        # Fit GP classifier
        X = data.x.cpu().numpy()
        Y = data.y.squeeze().cpu().numpy()
        Y = np.array([assign_class(y) for y in Y])
        gpc = GaussianProcessClassifier().fit(X, Y)

        # Generate random candiates, select the one with maximum label uncertainty
        domain = [config.bounds_design] * config.num_dim_design
        X_candidates = uniform_random_sample_domain(domain, 10000).numpy()
        X_candidates_prob = gpc.predict_proba(X_candidates)
        X_candidates_score = 1 - X_candidates_prob.max(axis=1)
        idx = np.argmax(X_candidates_score)
        next_x = torch.tensor([X_candidates[idx]])

        # End timer
        time_iter = time.time() - time_start

        # Query black box function, observe next_y
        with torch.no_grad():
            next_y = func(next_x)

        # Print
        print_iteration(iteration, next_x, next_y, None, time_iter)

        # Evaluate
        if eval_function:
            eval_metric, eval_data = eval_function(model_hes, config, data, next_x, iteration)
            eval_list.append(eval_metric)
            eval_data_list.append(eval_data)
            print(f'--\nEval metric: {eval_metric}')

        # Update training points
        data.x = torch.cat([data.x, next_x.detach()])
        data.y = torch.cat([data.y, next_y.detach()])

        # Re-initialize model for next iteration, use state_dict to speed up fitting
        mll_hes, model_hes = initialize_model(data, model_hes.state_dict())

        # Pickle trial info at each iteration (overwriting file from previous iteration)
        pickle_trial_info(config, data, eval_list, eval_data_list)

    # Optional final evaluation
    if final_eval_function:
        with torch.no_grad():
            final_eval_function(eval_list, config)