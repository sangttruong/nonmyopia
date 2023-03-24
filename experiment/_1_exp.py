import time
import torch
from models.actor import Actor
from experiment.checkpoint_manager import pickle_trial_info
from utils.utils import set_seed, generate_initial_data

def run(parms, env) -> None:

    set_seed(parms.seed)
    buffer = generate_initial_data(env, parms)
    actor = Actor(parms=parms, buffer=buffer)
    metrics = dict(
        eval_metric_list=[float("nan")] * parms.n_iterations,
        optimal_action_list=[float("nan")] * parms.n_iterations,
    )

    # Run BO loop
    for iteration in range(parms.start_iter, parms.start_iter + parms.n_iterations):    
        next_x = actor.query()
        next_y = env.func(next_x)

        # Update training points
        buffer.x = torch.cat([buffer.x, next_x.detach()])
        buffer.y = torch.cat([buffer.y, next_y.detach()])

        # Plot
        parms.plot_function(
            config=parms,
            iteation=iteration,
            buffer=buffer,
            previous_x=previous_x,
            next_x=next_x,
        )

        # Evaluate
        eval_metric, optimal_action = parms.eval_function(
            config=parms,
            buffer=buffer,
            iteration=iteration,
            next_x=next_x,
            previous_x=previous_x,
        )
        metrics["eval_metric_list"][iteration] = eval_metric
        metrics["optimal_action_list"][iteration] = optimal_action
        print(f"--\nEval metric: {eval_metric}")

        # Update previous_x
        previous_x = next_x

        # Re-train GP model
        actor.train(buffer)

        # Pickle trial info at each iteration (overwriting file from previous iteration)
        pickle_trial_info(parms, buffer, metrics)

    # Optional final evaluation
    with torch.no_grad():
        parms.final_eval_function(metrics["eval_metric_list"], parms)
