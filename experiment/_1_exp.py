import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from models.actor import Actor
from experiment.checkpoint_manager import pickle_trial_info
from utils.utils import set_seed, generate_initial_data
from utils.plot import draw_posterior, plot_topk
import matplotlib.pyplot as plt

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
    
    lookahead_steps = parms.lookahead_steps
    
    # Run BO loop
    for iteration in range(parms.n_iterations):    
        next_x, optimal_actions, eval_metric = actor.query(buffer, iteration)
        next_y = env.func(next_x)

        # Update training points
        buffer.x = torch.cat([buffer.x, next_x.detach().cpu()])
        buffer.y = torch.cat([buffer.y, next_y.detach().cpu()])
    
        # Evaluate
        eval_metric = eval_metric.cpu()
        optimal_actions = optimal_actions.cpu()
        
        # Plot optimal_action in special eval plot here
        # plot_topk(config=parms,
        #         env=env,
        #         buffer=buffer,
        #         iteration=iteration,
        #         next_x=buffer.x[-1],
        #         previous_x=buffer.x[-2],
        #         actions=optimal_actions)
        
        
        metrics["eval_metric_list"][iteration] = eval_metric.numpy().tolist()
        metrics["optimal_action_list"][iteration] = optimal_actions.numpy().tolist()
        print(f"Eval metric: {eval_metric.numpy().tolist()}")
        
        # Pickle trial info at each iteration (overwriting file from previous iteration)
        pickle_trial_info(parms, buffer, metrics, optimal_actions)

        # Fit the model
        WM = SingleTaskGP(buffer.x, buffer.y, covar_module=ScaleKernel(base_kernel=RBFKernel()))
        mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
        fit_gpytorch_model(mll)
        WM = WM.to(parms.device)
        
        if not parms.learn_hypers:
            WM = env.set_ground_truth_GP_hyperparameters(WM)

        # Draw posterior
        # draw_posterior(config=parms,
        #                model=WM,
        #                train_x=buffer.x,
        #                iteration=iteration)
        
        plt.figure(figsize=(7, 7))
        plt.vlines(buffer.x[-2], -5, 5, color='black', label='current location')
        plt.vlines(buffer.x[-2]-0.1, -5, 5, color='black', linestyle='--')
        plt.vlines(buffer.x[-2]+0.1, -5, 5, color='black', linestyle='--')
        plt.vlines(buffer.x[-1], -5, 5, color='red', label='optimal query')
        
        best_a = torch.max(optimal_actions.mean(0)).cpu().numpy()
        plt.vlines(best_a, -5, 5, color='blue', label='optimal action')

        train_x = torch.linspace(-1, 1, 100)
        train_y = env.func(train_x)
        plt.plot(
            train_x.cpu().numpy(), 
            train_y.cpu().numpy(), 
            'black', alpha=0.2, label='Ground truth')

        # compute posterior
        test_x = torch.linspace(-1, 1, 100).to(parms.device)
        posterior = WM.posterior(test_x)
        test_y = posterior.mean
        lower, upper = posterior.mvn.confidence_region()

        plt.plot(
            test_x.cpu().detach().numpy(), 
            test_y.cpu().detach().numpy(), 
            'green', label='Posterior mean')

        plt.fill_between(
            test_x.cpu().detach().numpy(), 
            lower.cpu().detach().numpy(), 
            upper.cpu().detach().numpy(), 
            alpha=0.25)
        
        # Scatter plot of all points using buffer.x and buffer.y with gradient color from red to blue indicating the order of point in list
        plt.scatter(buffer.x, buffer.y, c=range(len(buffer.x)), cmap='RdBu', marker='*', zorder=99)
        
        plt.tight_layout()
        plt.ylim(-5, 5)
        plt.legend()
        plt.savefig("test.png")
        plt.close()
        
        # Set WM to actor
        actor.set_WM(WM)
        
        # Adjust lookahead steps
        if lookahead_steps > 1 and iteration >= parms.lookahead_warmup:
            lookahead_steps -= 1
            actor.set_lookahead_steps(lookahead_steps)
        
    # Optional final evaluation
    with torch.no_grad():
        parms.final_eval_function(metrics["eval_metric_list"], parms)
