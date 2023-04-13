import torch
from argparse import ArgumentParser
from experiment._1_exp import run
from experiment._2_env import make as make_env
from experiment.checkpoint_manager import make_save_dir
from models.compute_expected_loss import (
    compute_expectedloss_topk,
    compute_expectedloss_minmax,
    compute_expectedloss_twoval,
    compute_expectedloss_mvs,
    compute_expectedloss_levelset,
    compute_expectedloss_multilevelset,
    compute_expectedloss_pbest,
    compute_expectedloss_bestofk,
)
from utils.plot import plot_topk, draw_metric
from utils.utils import eval_topk, splotlight_cost_function
import copy
from multiprocessing import Process, Manager
from threading import Thread
import dill as pickle
import numpy as np
import time

class Parameters:
    def __init__(self, args):
        # general arguments
        self.task = args.task
        self.set_task_parms()

        self.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        print(f"Using device {self.device}")
        self.gpu_id = args.gpu_id
        self.exp_id = args.exp_id
        self.mode = "train"
        self.check_dir = "experiments"
        self.save_dir = f"./results/exp_{self.exp_id:03d}"
        self.torch_dtype = torch.double

        self.algo = args.algo
        self.env_name = args.env_name        

        # Cost structure
        self.spotlight_cost_radius = None

        self.seed = args.seed
        self.seed_synthfunc = 1
        self.x_dim = 1
        self.y_dim = 1
        self.n_iterations = 2
        self.lookahead_steps = 10
        self.lookahead_warmup = self.n_iterations - self.lookahead_steps
        self.n_initial_points = 3
        self.local_init = True
        self.n_candidates = 1
        self.func_is_noisy = False
        self.func_noise = 0.1
        self.plot_iters = list(range(0, 101, 1))
        self.start_iter = 0

        # Algorithm parameters
        self.batch_size = 10
        self.lookahead_batch_sizes = [2] * self.lookahead_steps
        self.num_fantasies = [2] * self.lookahead_steps
        
        if self.algo == "hes":
            self.use_amortized_optimization = True 
        else:
            self.use_amortized_optimization = False 
            
        self.acq_opt_lr = 0.001 if self.use_amortized_optimization else 1e-3
        self.n_samples = 3
        self.decay_factor = 1

        # Optimizer
        # self.optimizer = "adam"
        self.acq_opt_iter = 5 if self.use_amortized_optimization else 3
        # self.acq_opt_iter = 500 if self.use_amortized_optimization else 3000   
        self.acq_warmup_iter = self.acq_opt_iter // 20
        self.acq_earlystop_iter = int(self.acq_opt_iter * 0.4)
        self.n_restarts = 1
        self.hidden_dim = 128

        # Amortization
        self.n_layers = 2
        self.activation = "elu"
        self.hidden_coeff = 4

        self.init_noise_thredhold = 0.01

        # Resampling
        """When n_resampling_max == 1 and n_resampling_improvement_threadhold is small, we have 
        the orange curve. n_resampling_max is large and n_resampling_improvement_threadhold is
        large, we have the pink curve (closer to stochastic gradient descent). We can interpolate
        between these 2 options by setting both hyperparameters to some moderate value. """
        self.n_resampling_max = 1
        self.n_resampling_improvement_threadhold = 0.01

        # Patients
        self.max_patient = 5000
        self.max_patient_resampling = 5

        # annealing for hes optimizer
        """When eta_min = acq_opt_lr, the learning rate is constant at acq_opt_lr
        large T_max corresponds to slow annealing
        """
        self.eta_min = 0.0001
        self.T_max = 100

        # Dataset parameters
        if self.env_name == "SynGP":
            self.n_obs = 50
            self.learn_hypers = False
            self.bounds = [-1, 1]
            
    def set_task_parms(self):
        if self.task == "topk":
            self.compute_expectedloss_function = compute_expectedloss_topk
            self.cost_function = splotlight_cost_function
            self.eval_function = eval_topk #TODO 
            self.final_eval_function = None #TODO
            self.plot_function = plot_topk #TODO
            self.n_actions = 1
            self.dist_weight = 1 # args.dist_weight
            self.dist_threshold = 0.5 # args.dist_threshold
            self.neighbor_size = 0.1
            self.epsilon = 1 # 1: no random reset, 0: random reset

        elif self.task == "minmax":
            self.compute_expectedloss_function = compute_expectedloss_minmax
            self.eval_function = None #TODO 
            self.final_eval_function = None #TODO
            self.plot_function = None #TODO
            self.n_actions = 2

        elif self.task == "twovalue":
            self.compute_expectedloss_function = compute_expectedloss_twoval
            self.eval_function = None #TODO 
            self.final_eval_function = None #TODO
            self.plot_function = None #TODO
            self.n_actions = None #TODO

        elif self.task == "mvs":
            self.compute_expectedloss_function = compute_expectedloss_mvs
            self.eval_function = None #TODO 
            self.final_eval_function = None #TODO
            self.plot_function = None #TODO
            self.n_actions = None #TODO
        
        elif self.task == "levelset":
            self.compute_expectedloss_function = compute_expectedloss_levelset
            self.eval_function = None #TODO 
            self.final_eval_function = None #TODO
            self.plot_function = None #TODO
            self.n_actions = None #TODO

        elif self.task == "multilevelset":
            self.compute_expectedloss_function = compute_expectedloss_multilevelset
            self.eval_function = None #TODO 
            self.final_eval_function = None #TODO
            self.plot_function = None #TODO
            self.n_actions = None #TODO

        elif self.task == "pbest":
            self.compute_expectedloss_function = compute_expectedloss_pbest
            self.eval_function = None #TODO 
            self.final_eval_function = None #TODO
            self.plot_function = None #TODO
            self.n_actions = None #TODO

        elif self.task == "bestofk":
            self.compute_expectedloss_function = compute_expectedloss_bestofk
            self.eval_function = None #TODO 
            self.final_eval_function = None #TODO
            self.plot_function = None #TODO
            self.n_actions = None #TODO

        else:
            raise NotImplemented


    def __str__(self):
        output = []
        for k in self.__dict__.keys():
            output.append(f"{k}: {self.__dict__[k]}")
        return "\n".join(output)


if __name__ == "__main__":

    # Parse args
    parser = ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[2,3])
    parser.add_argument("--task", type=str, default="topk")
    parser.add_argument("--env_name", type=str, default="SynGP")
    parser.add_argument("--algos", nargs="+", type=str, default=["HES"])
    parser.add_argument("--exp_id", type=int, default=0)
    parser.add_argument("--gpu_id", nargs="+", type=int)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    # with Manager() as manager:
        # metrics = manager.dict()
    metrics = {}
    list_processes = []

    for i, seed in enumerate(args.seeds):
        for j, algo in enumerate(args.algos):
            # Copy args
            local_args = copy.deepcopy(args)
            local_args.seed = seed
            local_args.gpu_id = args.gpu_id[i]
            local_args.algo = args.algos[j]
            
            local_parms = Parameters(local_args)
            
            # Make save dir
            make_save_dir(local_parms)

            # Init environment
            env = make_env(local_parms)
            
            # Run trials
            p = Thread(
                target=run,
                args=(local_parms, env, metrics),
            )
            list_processes.append(p)

    # Implement a simple queue system to run the experiments
    number_alive_processes = 0
    list_alive_processes = []
    for i, p in enumerate(list_processes):
        p.start()
        list_alive_processes.append(i)
        if len(list_alive_processes) >= args.n_jobs:
            while True:
                for j in list_alive_processes:
                    if not list_processes[j].is_alive():
                        list_alive_processes.remove(j)

                if len(list_alive_processes) < args.n_jobs:
                    break

                time.sleep(0.5)

    for pi in list_alive_processes:
        list_processes[pi].join()

    # Convert the metrics to a normal dict
    metrics = dict(metrics)

    # Draw regret curves
    list_metrics = []
    for i, algo in enumerate(args.algos):
        algo_metrics = []
        for i, seed in enumerate(args.seeds):
            algo_metrics.append(metrics[f"eval_metric_list_{algo}_{seed}"])
        list_metrics.append(algo_metrics)
    
    draw_metric(f"results", list_metrics, args.algos)
    
    # for trial in range(len(list_save_dir)):
    #     with open(f"{list_save_dir[trial]}/trial_info.pkl", "rb") as f:
    #         trial_info = pickle.load(f)
    #         metrics.append(trial_info.eval_metric_list)
            