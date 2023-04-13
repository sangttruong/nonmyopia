import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from models.actor import Actor
from experiment.checkpoint_manager import pickle_trial_info
from utils.utils import set_seed, generate_initial_data
from utils.plot import draw_posterior, plot_topk, draw_metric
import matplotlib.pyplot as plt

def run(parms, env, metrics) -> None:

    set_seed(parms.seed)
    buffer = generate_initial_data(env, parms)
    WM = SingleTaskGP(buffer.x, buffer.y, covar_module=ScaleKernel(base_kernel=RBFKernel()))
    mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
    fit_gpytorch_model(mll)
    WM = WM.to(parms.device)
    if not parms.learn_hypers:
        WM = env.set_ground_truth_GP_hyperparameters(WM)
        
    actor = Actor(parms=parms, WM=WM, buffer=buffer)

    metrics[f"eval_metric_list_{parms.algo}_{parms.seed}"] = [float("nan")] * parms.n_iterations
    metrics[f"optimal_action_list_{parms.algo}_{parms.seed}"] = [float("nan")] * parms.n_iterations
    
    lookahead_steps = parms.lookahead_steps
    
    # Run BO loop
    for iteration in range(parms.n_iterations):    
        next_x, optimal_actions, eval_metric = actor.query(buffer, iteration)
        next_y = env.func(next_x)

        # Update training points
        buffer.x = torch.cat([buffer.x, next_x])
        buffer.y = torch.cat([buffer.y, next_y])
    
        # Evaluate
        eval_metric = eval_metric.cpu().squeeze()
        optimal_actions = optimal_actions.cpu().squeeze()
        metrics[f"eval_metric_list_{parms.algo}_{parms.seed}"][iteration] = eval_metric.item()
        metrics[f"optimal_action_list_{parms.algo}_{parms.seed}"][iteration] = optimal_actions.numpy().tolist()
        print(f"Eval metric: {eval_metric.item()}")
        
        # Pickle trial info at each iteration (overwriting file from previous iteration)
        pickle_trial_info(parms, buffer, metrics[f"eval_metric_list_{parms.algo}_{parms.seed}"], metrics[f"optimal_action_list_{parms.algo}_{parms.seed}"])

        # Draw posterior
        draw_posterior(config=parms,
                       env=env,
                       model=WM,
                       buffer=buffer,
                       iteration=iteration,
                       optimal_actions=optimal_actions
        )
        
        # Plot optimal_action in special eval plot here
        if parms.x_dim == 2:
            plot_topk(config=parms,
                    env=env,
                    buffer=buffer,
                    iteration=iteration,
                    next_x=buffer.x[-1],
                    previous_x=buffer.x[-2],
                    actions=optimal_actions)
            
        # Fit the model
        WM = SingleTaskGP(buffer.x, buffer.y, covar_module=ScaleKernel(base_kernel=RBFKernel()))
        mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
        fit_gpytorch_model(mll)
        WM = WM.to(parms.device)
        
        if not parms.learn_hypers:
            WM = env.set_ground_truth_GP_hyperparameters(WM)
        
        # Set WM to actor
        actor.set_WM(WM)
        
        # Adjust lookahead steps
        if lookahead_steps > 1 and iteration >= parms.lookahead_warmup:
            lookahead_steps -= 1
            actor.set_lookahead_steps(lookahead_steps)
            
        
        # Draw eval metric
        # acqf_values = actor.eval()
        
        # draw_metric(save_dir=parms.save_dir,
        #             metrics=metrics[f"eval_metric_list_{parms.algo}_{parms.seed}"][:iteration+1]
        # )
        
        
    # Optional final evaluation
    # with torch.no_grad():
    #     parms.final_eval_function(metrics["eval_metric_list"], parms)
