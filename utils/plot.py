import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import copy
import torch 
from tueplots import bundles
plt.rcParams.update(bundles.iclr2023())

color={
    'C0':'#1f77b4', 
    'C1':'#ff7f0e', 
    'C2':'#2ca02c', 
    'C3':'#d62728', 
    'C4':'#9467bd', 
    'C5':'#bcbd22', 
    'C6':'#e377c2', 
    'C7':'#17becf', 
    # 'C8':'#bcbd22', 
    # 'C9':'#17becf',
}

figsize = [7, 7]
plt.rcParams["figure.figsize"] = figsize


def plot_synthfunc_2d(ax, env, config):
    """Plot synthetic function in 2d."""
    domain_plot = env.domain
    grid = 0.02
    xpts = np.arange(domain_plot[0][0], domain_plot[0][1], grid)
    ypts = np.arange(domain_plot[1][0], domain_plot[1][1], grid)
    X, Y = np.meshgrid(xpts, ypts)
    Z = env.vec(X, Y)
    cf = ax.contourf(X, Y, Z, 20, cmap=cm.GnBu, zorder=0)
    plot = ax.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
    cbar = plt.colorbar(cf, fraction=0.046, pad=0.04)
    # add colorbar here

def plot_function_contour(ax, env, config):
    plot_synthfunc_2d(ax, env, config)

def plot_data(ax, data):
    data_x = copy.deepcopy(data.x.cpu().detach()).numpy()
    for xi in data_x:
        ax.plot(xi[0], xi[1], "o", color="black", markersize=2)

def plot_next_query(ax, next_x):
    next_x = copy.deepcopy(next_x.cpu().detach()).numpy().reshape(-1)
    ax.plot(next_x[0], next_x[1], "o", color="deeppink", markersize=2)

def plot_settings(ax, env, config):
    bounds_plot = env.bounds
    bounds_plot_ext = [bounds_plot[0] - 0.05, bounds_plot[1] + 0.05]
    ax.set(
        xlabel="$x_1$", ylabel="$x_2$", xlim=bounds_plot_ext, ylim=bounds_plot_ext
    )

    # Set title
    if config.algo == "hes":
        title = "HES"
    if config.algo == "hes_vi":
        title = "HES VI"
    if config.algo == "hes_mc":
        title = "HES MC"
    if config.algo == "random":
        title = "Random"
    if config.algo == "qEI":
        title = "Expected Improvement"
    if config.algo == "qPI":
        title = "Probability of Improvement"
    if config.algo == "qSR":
        title = "Simple Regret"
    if config.algo == "qUCB":
        title = "Upper Confident Bound"
    if config.algo == "kg":
        title = "Knowledge Gradient"
    if config.algo == "rs":
        title = "Random Search"
    if config.algo == "us":
        title = "Uncertainty Sampling"
        
    # ax.set(title=title)
    ax.set_title(label=title, fontdict={"fontsize": 25})

def plot_action_samples(ax, action_samples, config):
    action_samples = copy.deepcopy(action_samples)
    action_samples = action_samples.reshape(
        -1, config.n_actions, config.x_dim
    )
    for x_actions in action_samples:
        lines2d = ax.plot(
            x_actions[0][0],
            x_actions[0][1],
            "v",
            color="b",
            alpha=0.5,
            markersize=4,
        )
        if config.n_actions >= 2:
            color = lines2d[0].get_color()
            ax.plot(
                x_actions[1][0],
                x_actions[1][1],
                "v",
                color="b",
                alpha=0.5,
                markersize=4,
            )
            line_1_x = [x_actions[0][0], x_actions[1][0]]
            line_1_y = [x_actions[0][1], x_actions[1][1]]
            ax.plot(line_1_x, line_1_y, "--", color=color)
        if config.n_actions >= 3:
            ax.plot(
                x_actions[2][0],
                x_actions[2][1],
                "v",
                color="b",
                alpha=0.5,
                markersize=4,
            )
            line_2_x = [x_actions[1][0], x_actions[2][0]]
            line_2_y = [x_actions[1][1], x_actions[2][1]]
            ax.plot(line_2_x, line_2_y, "--", color=color)

def plot_optimal_action(ax, optimal_action, config):
    optimal_action = optimal_action.reshape(-1, config.x_dim)
    ax.plot(
            optimal_action[:, 0],
            optimal_action[:, 1],
            "*",
            mfc="gold",
            mec="darkgoldenrod",
            markersize=4,
        )
        

def plot_groundtruth_optimal_action(ax, config):

    if config.n_actions <= 3:
        centers = [[4.0, 4.0], [2.45, 4.0], [4.0, 2.45]]
    elif config.n_actions == 5:
        centers = [[4.0, 4.0], [2.45, 4.0], [4.0, 2.45], [1.0, 4.0], [4.0, 1.0]]
    gt_optimal_action = np.array(centers)

    for x_action in gt_optimal_action:
        ax.plot(x_action[0], x_action[1], "s", color="blue", markersize=4)

def plot_spotlight(ax, config, previous_x):
    previous_x = previous_x.squeeze()
    previous_x = previous_x.cpu().detach().numpy()
    splotlight = plt.Rectangle(
        (previous_x[0] - config.neighbor_size, previous_x[1] - config.neighbor_size),
        2 * config.neighbor_size,
        2 * config.neighbor_size,
        color="b",
        fill=False,
    )
    ax.add_patch(splotlight)

def plot_topk(config, env, buffer, iteration, next_x, previous_x=None, actions=None, eval=False):
    """Plotting for topk."""
    print("Drawing topk...")
    if iteration in config.plot_iters:
        fig, ax = plt.subplots(figsize=(6, 6))

        plot_function_contour(ax, env, config)
        plot_data(ax, buffer)
        if actions is not None:
            plot_optimal_action(ax, actions, config)
        plot_next_query(ax, next_x)
        if config.neighbor_size and previous_x is not None:
            plot_spotlight(ax, config, previous_x)
        plot_settings(ax, env, config)

        # Save plot and close
        fig_name = f"{config.save_dir}/topk{'_eval' if eval else ''}_{iteration}.png"
        plt.savefig(fig_name, format="png")
        plt.close(fig)

def draw_posterior(config, env, model, buffer, iteration, optimal_actions=None):
    print("Drawing posterior...")
    if config.x_dim == 1:
        if optimal_actions is not None:
            optimal_actions = optimal_actions.reshape(-1, config.x_dim)
            best_a = optimal_actions.numpy()
            plt.vlines(best_a, -5, 5, color=color['C6'], label='optimal action', alpha=0.1)
        
        plt.vlines(buffer.x[-2], -5, 5, color='black', label='current location')
        plt.vlines(buffer.x[-2]-0.1, -5, 5, color='black', linestyle='--')
        plt.vlines(buffer.x[-2]+0.1, -5, 5, color='black', linestyle='--')
        plt.vlines(buffer.x[-1], -5, 5, color='red', label='optimal query')

        train_x = torch.linspace(-1, 1, 100)
        train_y = env.func(train_x)
        plt.plot(
            train_x.cpu().numpy(), 
            train_y.cpu().numpy(), 
            'black', alpha=0.2, label='Ground truth')

        # compute posterior
        test_x = torch.linspace(-1, 1, 100).to(config.device)
        posterior = model.posterior(test_x)
        test_y = posterior.mean
        lower, upper = posterior.mvn.confidence_region()

        plt.plot(
            test_x.cpu().detach().numpy(),
            test_y.cpu().detach().numpy(),
            color['C1'], label='Posterior mean')

        plt.fill_between(
            test_x.cpu().detach().numpy(),
            lower.cpu().detach().numpy(),
            upper.cpu().detach().numpy(), alpha=0.25,
            color=color['C2'])
        
        # Scatter plot of all points using buffer.x and buffer.y with gradient color from red to blue indicating the order of point in list
        plt.scatter(buffer.x, buffer.y, c=range(len(buffer.x)), cmap='Reds', marker='*', zorder=99)
        # plt.tight_layout()
        plt.ylim(-5, 5)
        
    elif config.x_dim == 2:
        grid = 20j
        test_x = np.mgrid[-1:1:grid, -1:1:grid].reshape(2,-1).T
        test_x = torch.tensor(test_x).to(config.device)
        
        posterior = model.posterior(test_x)
        
        test_y = posterior.mean.detach().cpu().numpy()
        xpts = np.linspace(-1, 1, int(abs(grid)))
        ypts = np.linspace(-1, 1, int(abs(grid)))
        X, Y = np.meshgrid(xpts, ypts)
        resol = int(abs(grid))
        Z = test_y.reshape(resol, resol).T
        cf = plt.contourf(X, Y, Z, 40, cmap=cm.coolwarm, zorder=0)
        cbar = plt.colorbar(cf, fraction=0.046, pad=0.04)
        plt.scatter(buffer.x[:, 0], buffer.x[:, 1], marker='*', color='black')
        
    else: 
        raise
    
    fig_name = f"{config.save_dir}/posterior_{iteration}.png"
    plt.savefig(fig_name, format="png")
    plt.close()
    
    
def draw_losses(config, losses, iteration):
    plt.plot(list(range(len(losses))), losses, label=f'Loss by iteration {iteration}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f"{config.save_dir}/losses_{iteration}.png", format="png")
    plt.close()
    
def draw_metric(save_dir, metrics, algos):
    if isinstance(metrics, list):
        metrics = np.array(metrics)
        
    plt.figure(figsize=(7, 7))
    for i, algo in enumerate(algos):
        mean = np.mean(metrics[i], axis=0)
        lower = np.min(metrics[i], axis=0)
        upper = np.max(metrics[i], axis=0)
        plt.plot(list(range(1, mean.shape[0]+1)), mean, label=algo)
        
        plt.fill_between(
            list(range(1, mean.shape[0]+1)),
            lower,
            upper, 
            alpha=0.25
        )
    
    plt.xlabel("Iteration")
    plt.ylabel("Eval metric")
    fig_name = f"{save_dir}/eval_metric.png"
    plt.savefig(fig_name, format="png")
    plt.close()