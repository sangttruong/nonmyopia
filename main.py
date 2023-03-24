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


class Parameters:
    def __init__(self, args):
        # general arguments
        self.task = args.task
        self.set_task_parms()

        self.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
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

        self.seed = 11
        self.seed_synthfunc = 1
        self.x_dim = 2
        self.y_dim = 1
        self.lookahead_steps = 10
        self.n_initial_points = 10
        self.local_init = True
        self.n_iterations = 100
        self.n_candidates = 1
        self.func_is_noisy = False
        self.func_noise = 0.1
        self.plot_iters = list(range(0, 101, 1))
        self.start_iter = 0

        # Algorithm parameters
        self.batch_size = 10
        self.lookahead_batch_sizes = [2] * self.lookahead_steps
        self.num_fantasies = [2] * self.lookahead_steps
        self.use_amortized_optimization = True
        self.acq_opt_lr = 0.001 if self.use_amortized_optimization else 0.1
        self.n_samples = 1
        self.decay_factor = 1

        # optimizer
        self.optimizer = "adam"
        self.acq_opt_iter = 100  # 1000
        self.n_restarts = 1
        self.hidden_dim = 32

        # amortization
        self.n_layers = 2
        self.activation = "elu"
        self.hidden_coeff = 4

        self.init_noise_thredhold = 0.01

        # resampling
        """When n_resampling_max == 1 and n_resampling_improvement_threadhold is small, we have 
        the orange curve. n_resampling_max is large and n_resampling_improvement_threadhold is
        large, we have the pink curve (closer to stochastic gradient descent). We can interpolate
        between these 2 options by setting both hyperparameters to some moderate value. """
        self.n_resampling_max = 1
        self.n_resampling_improvement_threadhold = 0.01

        # patients
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
            
    def set_task_parms(self):
        if self.task == "topk":
            self.compute_expectedloss_function = compute_expectedloss_topk
            self.eval_function = None #TODO 
            self.final_eval_function = None #TODO
            self.plot_function = None #TODO
            self.n_actions = 3
            self.dist_weight = 1 # args.dist_weight
            self.dist_threshold = 0.5 # args.dist_threshold


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
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--task", type=str, default="topk")
    parser.add_argument("--env_name", type=str, default="SynGP")
    parser.add_argument("--algo", nargs="+", type=str, default="hes")
    parser.add_argument("--exp_id", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    # Init params
    parms = Parameters(args)

    # Make save dir
    make_save_dir(parms)

    # Init environment
    env = make_env(parms)

    # Run hes trial
    run(
        parms=parms,
        env=env
    )
