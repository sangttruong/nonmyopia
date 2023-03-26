import time
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from models.actor import Actor
from experiment.checkpoint_manager import pickle_trial_info
from utils.utils import set_seed, generate_initial_data

def run(parms, env) -> None:

    set_seed(parms.seed)
    buffer = generate_initial_data(env, parms)
    WM = SingleTaskGP(buffer.x, buffer.y, covar_module=ScaleKernel(base_kernel=RBFKernel()))
    actor = Actor(parms=parms, WM=WM, buffer=buffer)
    mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
    fit_gpytorch_model(mll)
    WM = WM.to(parms.device)

    metrics = dict(
        eval_metric_list=[float("nan")] * parms.n_iterations,
        optimal_action_list=[float("nan")] * parms.n_iterations,
    )
    
    previous_x = buffer.x[-parms.n_restarts:]
    previous_y = buffer.y[-parms.n_restarts:]

    # Run BO loop
    for iteration in range(parms.start_iter, parms.start_iter + parms.n_iterations):    
        next_x = actor.query((previous_x, previous_y))
        next_y = env.func(next_x)

        # Update training points
        buffer.x = torch.cat([buffer.x, next_x.detach().cpu()])
        buffer.y = torch.cat([buffer.y, next_y.detach().cpu()])

        # Evaluate
        eval_metric, optimal_actions = parms.eval_function(
            config=parms,
            env=env,
            actor=actor,
            buffer=buffer,
            iteration=iteration
        )
        
        metrics["eval_metric_list"][iteration] = eval_metric
        metrics["optimal_action_list"][iteration] = optimal_actions
        print(f"Eval metric: {eval_metric}")

        # Update previous_x and previous_y
        previous_x = buffer.x[-parms.n_restarts:]
        previous_y = buffer.y[-parms.n_restarts:]
        
        # Pickle trial info at each iteration (overwriting file from previous iteration)
        pickle_trial_info(parms, buffer, metrics, optimal_actions)

        # Fit the model
        WM = SingleTaskGP(buffer.x, buffer.y, covar_module=ScaleKernel(base_kernel=RBFKernel()))
        mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
        fit_gpytorch_model(mll)
        WM = WM.to(parms.device)
        
        if not parms.learn_hypers:
            WM = env.set_ground_truth_GP_hyperparameters(WM)

        # Set WM to actor
        actor.set_WM(WM)
        
    # Optional final evaluation
    with torch.no_grad():
        parms.final_eval_function(metrics["eval_metric_list"], parms)
