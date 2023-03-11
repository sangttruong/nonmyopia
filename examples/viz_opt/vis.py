import matplotlib.pyplot as plt
import os, torch, dill as pickle, numpy as np

from argparse import Namespace, ArgumentParser
parser = ArgumentParser()
# general arguments
parser.add_argument("--plot_dir", type=str)
args = parser.parse_args()


plt.figure(figsize=(7, 7))

# algos = [
#     "hes_vi", 
#     "hes_mc", 
#     "random", 
#     "qEI", 
#     "qPI", 
#     "qSR", 
#     "qUCB", 
#     #"ddpg"
# ]

algos = os.listdir(f'{args.plot_dir}')

for algo in algos:
    if algo.startswith('.'): continue
    func_value_of_one_algo_diff_seed = []
    folder_names_of_an_algo = os.listdir(f'{args.plot_dir}/{algo}')
    # >>> Expecting several seed for each algorithm
    for check_dir in folder_names_of_an_algo:
        if check_dir.startswith('.'): continue
        # Unpickle trial_info Namespace
        with open(f"{args.plot_dir}/{algo}/{check_dir}/trial_info.pkl", "rb") as file:
            trial_info = pickle.load(file)

        c = trial_info.config
        sf = c.func
        def func(x_list):
            """Synthetic function with torch tensor input/output."""
            x_list = [xi.cpu().detach().numpy().tolist() for xi in x_list]
            x_list = np.array(x_list).reshape(-1, c.n_dim_design)
            y_list = [sf(x) for x in x_list]
            y_list = y_list[0] if len(y_list) == 1 else y_list
            y_tensor = torch.tensor(np.array(y_list).reshape(-1, 1))
            return y_tensor

        optimal_action_list = trial_info.optimal_action_list
        optimal_action_list = [torch.tensor(i) for i in optimal_action_list]
        optimal_action_list = torch.stack(optimal_action_list)
        list_of_func_value_at_optimal_action = func(optimal_action_list)
        list_of_func_value_at_optimal_action = list_of_func_value_at_optimal_action.squeeze()
        # >>> Function value of optimal action for each BO iteration 
        # >>> for a given seed, for a given algorithm
        func_value_of_one_algo_diff_seed.append(list_of_func_value_at_optimal_action)

    func_value_of_one_algo_diff_seed = torch.stack(func_value_of_one_algo_diff_seed)
    func_value_one_algo_mean = func_value_of_one_algo_diff_seed.mean(dim=0)
    func_value_one_algo_std = (func_value_of_one_algo_diff_seed.var(dim=0))**0.5

    iteration = np.arange(len(func_value_one_algo_mean))
    plt.plot(iteration, func_value_one_algo_mean, label=algo, c='b')
    plt.fill_between(
        iteration,
        func_value_one_algo_mean-func_value_one_algo_std,
        func_value_one_algo_mean+func_value_one_algo_std,
        alpha=0.2,
    )
    
plt.legend()
plt.savefig(f"{args.plot_dir}/func_optimal_output.pdf", bbox_inches="tight")
