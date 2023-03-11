#!/usr/bin/env python
""""""

from tqdm import tqdm
import math
from sklearn.gaussian_process import GaussianProcessClassifier
from .mykg import MyqKnowledgeGradient, initialize_action_tensor_kg
from .hentropy import HEntropySearch
from botorch.optim import optimize_acqf
from botorch.acquisition import (
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
import os
from pathlib import Path
from argparse import Namespace

__author__ = ""
__copyright__ = "Copyright 2022, Stanford University"

import time
import torch
import dill as pickle
import numpy as np
import matplotlib.pyplot as plt


def uniform_random_sample_domain(domain, n, config):
    """Draws a sample uniformly at random from domain (a list of tuple bounds)."""
    list_of_arr_per_dim = [np.random.uniform(
        dom[0], dom[1], n) for dom in domain]
    return torch.tensor(
        np.array(
            list_of_arr_per_dim).T, device=config.device, dtype=config.torch_dtype
    )


def generate_initial_data(func, config):
    ngen = config.n_initial_points
    domain = [config.bounds_design] * config.n_dim_design
    data_x = uniform_random_sample_domain(domain, ngen, config)  # n x dim
    data_y = func(data_x)  # n x 1
    if config.func_is_noisy:
        data_y = data_y + config.func_noise * torch.randn_like(
            data_y, config.device, config.torch_dtype
        )
    data = Namespace(x=data_x, y=data_y)
    return data


def initialize_model(data, state_dict=None, covar_module=None):
    model = SingleTaskGP(data.x, data.y, covar_module=covar_module).to(data.x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def print_iteration(next_x, next_y, time_iter):
    """Print information at each iteration of HES."""
    print(f"next_x = {next_x.cpu().detach().numpy().tolist()}")
    print(f"next_y = {next_y.cpu().detach().numpy().tolist()}")
    # print(f"action_samples = {action_samples.cpu().detach().numpy().tolist()}")
    print(f"time_iter = {time_iter}.")


def set_random_seed_at_start(seed):
    """Set random seed at start of run."""
    torch.manual_seed(seed=seed)
    np.random.seed(seed)


def set_random_seed_at_iteration(seed, iteration):
    """
    Set random seed at a given iteration, using seed and iteration (both positive
    integers) as inputs.
    """
    # First set initial random seed
    torch.manual_seed(seed=seed)
    np.random.seed(seed)

    # Then multiply iteration with a random integer and set as new seed
    new_seed = iteration * np.random.randint(1e6)
    torch.manual_seed(seed=new_seed)
    np.random.seed(new_seed)


def print_model_hypers(model):
    """Print current hyperparameters of GP model."""
    raw_hypers_str = (
        "\n*Raw GP hypers: "
        f"\nmodel.covar_module.base_kernel.raw_lengthscale={model.covar_module.base_kernel.raw_lengthscale.tolist()}"
        f"\nmodel.covar_module.raw_outputscale={model.covar_module.raw_outputscale.tolist()}"
        f"\nmodel.likelihood.noise_covar.raw_noise={model.likelihood.noise_covar.raw_noise.tolist()}"
    )
    actual_hypers_str = (
        "\n*Actual GP hypers: "
        f"\nmodel.covar_module.base_kernel.lengthscale={model.covar_module.base_kernel.lengthscale.tolist()}"
        f"\nmodel.covar_module.outputscale={model.covar_module.outputscale.tolist()}"
        f"\nmodel.likelihood.noise_covar.noise={model.likelihood.noise_covar.noise.tolist()}"
    )
    print(raw_hypers_str)
    print(actual_hypers_str + "\n")


def pickle_trial_info(config, data, eval_metric_list, optimal_action_list):
    """Save trial info as a pickle in directory specified by config."""
    # Build trial info Namespace
    data = Namespace(x=data.x.cpu().detach().numpy(),
                     y=data.y.cpu().detach().numpy())
    trial_info = Namespace(
        config=config, 
        data=data, 
        eval_metric_list=eval_metric_list, 
        optimal_action_list=optimal_action_list
    )

    # Pickle trial info
    dir_path = Path(str(config.save_dir))
    file_path = dir_path / "trial_info.pkl"
    with open(str(file_path), "wb") as file_handle:
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
            dir_path = Path(str(init_dir_path) + "_" + str(i).zfill(2))

    config.save_dir = str(dir_path)
    print(f"Created save_dir: {config.save_dir}")


def optimize_rs(config):
    """Optimize random search (rs) acquisition function, return next_x."""
    domain = [config.bounds_design] * config.n_dim_design
    data_x = uniform_random_sample_domain(domain, 1)
    next_x = data_x[0].reshape(1, -1)
    return next_x


def optimize_us(model, config):
    """Optimize uncertainty sampling (us) acquisition function, return next_x."""
    domain = [config.bounds_design] * config.n_dim_design
    n_acq_opt_samp = 500
    data_x = uniform_random_sample_domain(domain, n_acq_opt_samp)
    acq_values = model(data_x).variance
    best = torch.argmax(acq_values.view(-1), dim=0)
    next_x = data_x[best].reshape(1, -1)
    return next_x


def optimize_kg(batch_x0s, batch_a1s, model, sampler, config, iteration):
    """Optimize knowledge gradient (kg) acquisition function, return next_x."""
    if not config.n_dim_design == config.n_dim_action:
        batch_a1s = initialize_action_tensor_kg(config)

    qkg = MyqKnowledgeGradient(model, config, sampler)

    optimizer = torch.optim.Adam([batch_x0s, batch_a1s], lr=config.acq_opt_lr)
    for i in range(config.acq_opt_iter):
        losses = -qkg(batch_x0s, batch_a1s)
        loss = losses.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_x0s.data.clamp_(config.bounds_design[0], config.bounds_design[1])
        batch_a1s.data.clamp_(config.bounds_action[0], config.bounds_action[1])
        if (i + 1) % (config.acq_opt_iter // 5) == 0 or i == config.acq_opt_iter - 1:
            print(iteration, i + 1, loss.item())
    acq_values = qkg(batch_x0s, batch_a1s)
    best = torch.argmax(acq_values.view(-1), dim=0)
    next_x = batch_x0s[best]
    return next_x


def optimize_hes(hes, iteration):
    c = hes.config
    params = []
    if c.algo == "hes_vi":
        params_ = [
            hes.maps_i[i][j].parameters()
            for j in range(c.n_restarts)
            for i in range(c.lookahead_steps + 1)
        ]

        for i in range(len(params_)):
            params += list(params_[i])
    elif c.algo == "hes_mc":
        for i in range(c.lookahead_steps + 1):
            params.append(hes.po[i])

    # Test number of params in VI or MC
    if iteration == c.start_iter:
        total_num_params = sum(p.numel() for p in params if p.requires_grad)
        print(f"Total params: {total_num_params}")
        if c.algo == "hes_vi":
            nn_params = 0
            for i in range(0, c.lookahead_steps + 1):
                nn_params += sum(p.numel()
                                 for p in hes.maps_i[i][0].parameters())
            assert c.n_restarts * nn_params == total_num_params
        elif c.algo == "hes_mc":
            params_A = 0
            params_B = 0
            for i in range(c.lookahead_steps + 1):
                if i == 0 or i == 1:
                    params_A += c.n_samples ** i
                else:
                    tmp = c.n_samples
                    for j in range(1, i):
                        tmp *= math.ceil(c.n_samples / (c.decay_factor ** j))
                    if i == c.lookahead_steps:
                        params_B = c.n_dim_action * c.n_actions * tmp
                    else:
                        params_A += tmp
            params_A = c.n_dim_design * params_A
            assert c.n_restarts * (params_A + params_B) == total_num_params

    optim = torch.optim.Adam(params, lr=c.acq_opt_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=c.T_max, eta_min=c.eta_min
    )

    losses = []
    lrs = []
    patient = c.max_patient
    min_loss = float("inf")
    print("start optimizing acquisition function")
    for _ in tqdm(range(c.acq_opt_iter)):
        loss = -hes().sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        if c.algo == "hes_mc":
            for i in range(c.lookahead_steps + 1):
                hes.po[i] = torch.tanh(params[i])

        lrs.append(scheduler.get_last_lr())
        losses.append(loss.cpu().detach().numpy())

        if loss < min_loss:
            min_loss = loss
            patient = c.max_patient
            acq_values = hes()  # get acq values for all restarts
            # best result from all restarts
            best_restart = torch.argmax(acq_values)
            next_x = hes.po[0][best_restart]

        else:
            patient -= 1

        if patient < 0:
            break

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(np.array(losses), "b-", linewidth=1)
    ax2.plot(np.array(lrs), "r-", linewidth=1)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="b")
    ax2.set_ylabel("Learning rate", color="r")
    if not os.path.exists(f"{c.save_dir}/{c.algo}"):
        os.makedirs(f"{c.save_dir}/{c.algo}")
    plt.savefig(f"{c.save_dir}/{c.algo}/acq_opt{iteration}.png",
                bbox_inches="tight")

    return min_loss, next_x


def run_hes_trial(
    func,
    config,
    plot_function=None,
    eval_function=None,
    final_eval_function=None,
) -> None:
    """Run a single trial of H-Entropy Search using inputs as configuration."""

    c = config
    set_random_seed_at_start(config.seed)

    # Set initial observations
    if config.init_data is None:
        with torch.no_grad():
            config.init_data = generate_initial_data(func, config)
    data = config.init_data
    previous_x = data.x[-1]

    # Initialize model
    mll_hes, model_hes = initialize_model(
        data, covar_module=ScaleKernel(base_kernel=RBFKernel())
    )

    # Fit the model
    if not config.learn_hypers:
        print(
            f"config.learn_hypers={config.learn_hypers}, using hypers from config.hypers"
        )
        model_hes.covar_module.base_kernel.lengthscale = [
            [config.hypers["ls"]]]
        # NOTE: GPyTorch outputscale should be set to the SynthFunc alpha squared
        model_hes.covar_module.outputscale = config.hypers["alpha"] ** 2
        model_hes.likelihood.noise_covar.noise = [config.hypers["sigma"]]

        model_hes.covar_module.base_kernel.raw_lengthscale.requires_grad_(
            False)
        model_hes.covar_module.raw_outputscale.requires_grad_(False)
        model_hes.likelihood.noise_covar.raw_noise.requires_grad_(False)

    fit_gpytorch_model(mll_hes)
    print_model_hypers(model_hes)

    # Logging
    eval_metric_list = []
    # >>> list of evaluation metrics (related to acquisition value)
    # >>> for each iteration
    optimal_action_list = []
    # >>> list of optimal action for each iteration

    # Run BO loop
    for iteration in range(config.start_iter, config.start_iter + config.n_iterations):
        # Reset random seed at start of each iteration (still based on config.seed)
        set_random_seed_at_iteration(config.seed, iteration)

        print("\n" + "---" * 5 + f" Iteration {iteration} " + "---" * 5)
        time_start = time.time()
        if c.algo in ["hes_vi", "hes_mc"]:
            resampling_step = 0
            patient = config.max_patient_resampling
            min_loss = float("inf")
            while resampling_step < config.n_resampling_max and patient > 0:
                # Define hes object
                hes = HEntropySearch(config, model_hes, data=data)

                # Optimize acquisition function
                if config.algo in ["hes_mc", "hes_vi"]:
                    min_loss_, next_x = optimize_hes(hes, iteration)
                elif config.algo == "rs":
                    next_x = optimize_rs(config)
                elif config.algo == "us":
                    next_x = optimize_us(model_hes, config)
                elif config.algo == "kg" or config.algo == "kgtopk":
                    raise NotImplemented

                if min_loss_ < min_loss:
                    min_loss = min_loss_
                    patient = config.max_patient_resampling
                    next_y = func(next_x).detach()
                else:
                    patient -= 1

                print(
                    f"Resampling step {resampling_step}, Objective: {min_loss}")
                resampling_step += 1

        elif c.algo == "random":
            next_x = c.bounds_design[0] + (
                c.bounds_design[1] - c.bounds_design[0]
            ) * torch.rand([c.n_candidates, c.n_dim_design], device=c.device)
            next_y = func(next_x).detach()

        elif c.algo in ["qEI", "qPI", "qSR", "qUCB"]:
            sampler = SobolQMCNormalSampler(
                num_samples=c.n_samples, seed=0, resample=False
            )
            if c.algo == "qEI":
                acq_function = qExpectedImprovement(
                    model_hes, best_f=data.y.max(), sampler=sampler
                )
            elif c.algo == "qPI":
                acq_function = qProbabilityOfImprovement(
                    model_hes, best_f=data.y.max(), sampler=sampler
                )
            elif c.algo == "qSR":
                acq_function = qSimpleRegret(model_hes, sampler=sampler)
            elif c.algo == "qUCB":
                acq_function = qUpperConfidenceBound(
                    model_hes, beta=0.1, sampler=sampler
                )

            # to keep the restart conditions the same
            torch.manual_seed(seed=0)
            bounds = torch.tensor(
                [
                    [c.bounds_design[0]] * c.n_dim_design,
                    [c.bounds_design[1]] * c.n_dim_design,
                ]
            ).to(config.device).double()
            next_x, _ = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=c.n_restarts,
                raw_samples=1000,
                options={},
            )
            next_y = func(next_x).detach()

        # End timer
        time_iter = time.time() - time_start

        # Print
        print_iteration(next_x, next_y, time_iter)

        # Plot
        if plot_function:
            plot_function(
                config=config,
                iteation=iteration,
                data=data,
                previous_x=previous_x,
                next_x=next_x,
            )

        # Update training points
        data.x = torch.cat([data.x, next_x.detach()])
        data.y = torch.cat([data.y, next_y])

        # Evaluate
        if eval_function:
            eval_metric, optimal_action = eval_function(
                config=config,
                data=data,
                iteration=iteration,
                next_x=next_x,
                previous_x=previous_x,
            )
            eval_metric_list.append(eval_metric)
            optimal_action_list.append(optimal_action)
            print(f"--\nEval metric: {eval_metric}")

        # Update previous_x
        previous_x = next_x

        # Re-initialize model for next iteration, use state_dict to speed up fitting
        mll_hes, model_hes = initialize_model(
            data,
            model_hes.state_dict(),
            covar_module=ScaleKernel(base_kernel=RBFKernel()),
        )

        # Pickle trial info at each iteration (overwriting file from previous iteration)
        pickle_trial_info(config, data, eval_metric_list, optimal_action_list)

    # Optional final evaluation
    if final_eval_function:
        with torch.no_grad():
            final_eval_function(eval_metric_list, config)


def run_gpclassifier_trial(
    func,
    config,
    plot_function=None,
    eval_function=None,
    final_eval_function=None,
):
    """Run a single trial of GP classifier using inputs as configuration."""

    assert config.algo == "gpclassifier" and (
        config.app == "levelset" or config.app == "multilevelset"
    )

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
        if config.app == "levelset":
            return 1 if y > config.levelset_threshold else 0
        else:
            label = 0
            for threshold in config.levelset_thresholds:
                if y < threshold:
                    return label
                label += 1
            return label

    # Logging
    eval_metric_list = []
    optimal_action_list = []

    # Run BO loop
    for iteration in range(1, config.n_iteration + 1):

        print("---" * 5 + f" Iteration {iteration} " + "---" * 5)
        time_start = time.time()

        # Fit the model
        fit_gpytorch_model(mll_hes)

        # Fit GP classifier
        X = data.x.cpu().numpy()
        Y = data.y.squeeze().cpu().numpy()
        Y = np.array([assign_class(y) for y in Y])
        gpc = GaussianProcessClassifier().fit(X, Y)

        # Generate random candiates, select the one with maximum label uncertainty
        domain = [config.bounds_design] * config.n_dim_design
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
            eval_metric, optimal_action = eval_function(
                model_hes, config, data, next_x, iteration
            )
            eval_metric_list.append(eval_metric)
            optimal_action_list.append(optimal_action)
            print(f"--\nEval metric: {eval_metric}")

        # Update training points
        data.x = torch.cat([data.x, next_x.detach()])
        data.y = torch.cat([data.y, next_y.detach()])

        # Re-initialize model for next iteration, use state_dict to speed up fitting
        mll_hes, model_hes = initialize_model(data, model_hes.state_dict())

        # Pickle trial info at each iteration (overwriting file from previous iteration)
        pickle_trial_info(config, data, eval_metric_list, optimal_action_list)

    # Optional final evaluation
    if final_eval_function:
        with torch.no_grad():
            final_eval_function(eval_metric_list, config)
