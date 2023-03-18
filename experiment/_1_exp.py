from models.ehig_acqf import EHIG
from argparse import Namespace
import time
import torch
import numpy as np
from models.actor import Actor


def uniform_random_sample_domain(domain, n, config):
    """Draws a sample uniformly at random from domain (a list of tuple bounds)."""
    list_of_arr_per_dim = [np.random.uniform(
        dom[0], dom[1], n) for dom in domain]
    return torch.tensor(
        np.array(
            list_of_arr_per_dim).T, device=config.device, dtype=config.torch_dtype
    )


def generate_initial_data(env, config):
    data_x = uniform_random_sample_domain(
        config.domain, 
        config.n_initial_points, 
        config
        )  # n x dim
    data_y = env.func(data_x)  # n x 1
    if config.func_is_noisy:
        data_y = data_y + config.func_noise * torch.randn_like(
            data_y, config.device, config.torch_dtype
        )
    data = Namespace(x=data_x, y=data_y)
    return data


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


def run_hes_trial(
    parms,
    env,
    initial_data,
    plot_function=None,
    eval_function=None,
    final_eval_function=None,
) -> None:
    """Run a single trial of H-Entropy Search using inputs as configuration."""
    # Set up the environment
    set_random_seed_at_start(parms.seed)

    # Set initial observations
    if initial_data is None:
        with torch.no_grad():
            data = generate_initial_data(env, parms)
    else: 
        data = initial_data
        
    previous_x = data.x[-1]

    actor = Actor(parms)

    # Logging
    eval_metric_list = []
    # >>> list of evaluation metrics (related to acquisition value)
    # >>> for each iteration
    optimal_action_list = []
    # >>> list of optimal action for each iteration

    # Run BO loop
    for iteration in range(parms.start_iter, parms.start_iter + parms.n_iterations):
        # Reset random seed at start of each iteration (still based on config.seed)
        set_random_seed_at_iteration(parms.seed, iteration)

        print("\n" + "---" * 5 + f" Iteration {iteration} " + "---" * 5)
        time_start = time.time()
        if parms.algo in ["hes_vi", "hes_mc"]:
            resampling_step = 0
            patient = parms.max_patient_resampling
            min_loss = float("inf")
            while resampling_step < parms.n_resampling_max and patient > 0:
                # Define hes object
                hes = EHIG(parms, model_hes, data=data)

                # Optimize acquisition function
                next_x = actor.query()

                if min_loss_ < min_loss:
                    min_loss = min_loss_
                    patient = parms.max_patient_resampling
                    next_y = env.func(next_x).detach()
                else:
                    patient -= 1

                print(
                    f"Resampling step {resampling_step}, Objective: {min_loss}")
                resampling_step += 1

        next_x = actor.query()
        next_y = env.func(next_x).detach()

        # Update training points
        data.x = torch.cat([data.x, next_x.detach()])
        data.y = torch.cat([data.y, next_y])

        # End timer
        time_iter = time.time() - time_start

        # Print
        print_iteration(next_x, next_y, time_iter)

        # Plot
        if plot_function:
            plot_function(
                config=parms,
                iteation=iteration,
                data=data,
                previous_x=previous_x,
                next_x=next_x,
            )

        # Evaluate
        if eval_function:
            eval_metric, optimal_action = eval_function(
                config=parms,
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
        pickle_trial_info(parms, data, eval_metric_list, optimal_action_list)

    # Optional final evaluation
    if final_eval_function:
        with torch.no_grad():
            final_eval_function(eval_metric_list, parms)
