import os
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from botorch import fit_gpytorch_model
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.kernels import RBFKernel, ScaleKernel
from experiment._1_exp import initialize_model, make_save_dir, run_hes_trial
from experiment._2_env import make as make_env
from utils.utilities import get_init_data
from utils.plot import plot_topk
from models.compute_expected_loss import (
    compute_expectedloss_topk, 
    compute_expectedloss_minmax,
    compute_expectedloss_twovalue,
    compute_expectedloss_mvs,
    compute_expectedloss_levelset,
    compute_expectedloss_multilevelset,
    compute_expectedloss_pbest,
    compute_expectedloss_bestofk,
)


class Parameters:
    def __init__(self, args):
        # general arguments
        self.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        self.app = args.app

        if self.app == "topk":
            self.compute_expectedloss_function = compute_expectedloss_topk
        elif self.app == "minmax"
            # TODO ... 
            
        else:
            raise NotImplemented
        
        self.algo = args.algo
        self.dataset = args.dataset
        self.dist_weight = args.dist_weight
        self.dist_threshold = args.dist_threshold
        self.exp_id = args.exp_id
        
        self.seed = 11
        self.seed_synthfunc = 1
        self.n_dim = 2
        self.n_actions = 1
        self.gpuid = 0
        self.lookahead_steps = 1
        self.n_initial_points = 10
        self.r = None
        self.local_init = True
        self.n_iterations = 100
        self.n_candidates = 1
        self.func_is_noisy = False
        self.func_noise = 0.1
        self.plot_iters = list(range(0, 101, 1))
        self.start_iter = 0
        
        self.torch_dtype = torch.double

        # Algorithm parameters
        if self.algo == "hes_vi":
            self.vi = True
            self.acq_opt_lr = 0.001
        elif self.algo == "hes_mc":
            self.vi = False
            self.acq_opt_lr = 0.1
        else:
            raise NotImplementedError
        
        # MC approximation
        self.n_samples = 16
        self.decay_factor = 1

        # optimizer
        self.optimizer = "adam"
        self.acq_opt_iter = 100  # 1000
        self.n_restarts = 128

        # amortization
        self.n_layers = 2
        self.activation = "elu"
        self.hidden_coeff = 4

        # baseline
        self.baseline = False
        self.baseline_n_layers = 2
        self.baseline_hidden_coeff = 1
        self.baseline_activation = "relu"
        self.baseline_lr = 0.1

        self.init_noise_thredhold = 0.01

        # resampling
        self.n_resampling_max = 1
        self.n_resampling_improvement_threadhold = 0.01
        """When n_resampling_max == 1 and n_resampling_improvement_threadhold is small, we have 
        the orange curve. n_resampling_max is large and n_resampling_improvement_threadhold is
        large, we have the pink curve (closer to stochastic gradient descent). We can interpolate
        between these 2 options by setting both hyperparameters to some moderate value. """

        # gp hyperparameters
        self.learn_hypers = False

        # patients
        self.max_patient = 5000
        self.max_patient_resampling = 5

        # annealing for hes optimizer
        self.eta_min = 0.0001
        """When eta_min = acq_opt_lr, the learning rate is constant at acq_opt_lr"""
        self.T_max = 100
        """large T_max corresponds to slow annealing"""

        # Check parameter
        self.mode = "train"
        self.check_dir = "experiments"

        # Fix arg types, set defaults, perform checks
        assert self.lookahead_steps > 0

        # Initialize synthetic function
        if self.dataset == "synthetic":
            hypers = {"ls": 0.1, "alpha": 2.0, "sigma": 1e-2, "n_dimx": self.n_dim}
            self.hypers = hypers if not self.learn_hypers else None

            # TODO: need to figure out the bound for input in chemical dataset
            self.bounds = [-1, 1]
            self.domain = [self.bounds] * self.n_dim
            self.n_obs = 50
        
        self.save_dir = f"./results/exp_{self.exp_id:03d}"

    # def store(self):
    #     # Store attributes of this class in a json file in the save directory
    #     json_parms = json.dumps(self.__dict__, indent=4)
    #     with open(os.path.join(self.save_dir, "parameters.json"), "w") as f:
    #         f.write(json_parms)  # write json to file
            
def eval_topk(config, data, iteration, next_x, previous_x):
    """Return evaluation metric."""

    c = config
    noise = torch.rand([c.n_restarts, c.n_actions, c.n_dim], device=c.device)
    bayes_actions = (
        c.bounds[0] + (c.bounds[1] - c.bounds[0]) * noise
    )
    bayes_actions.requires_grad_(True)
    sampler = SobolQMCNormalSampler(
        sample_shape=c.n_samples, resample=False, collapse_batch_dims=True
    )
    mll_hes, model_hes = initialize_model(
        data, covar_module=ScaleKernel(base_kernel=RBFKernel())
    )

    if not config.learn_hypers:
        print(
            f"config.learn_hypers={config.learn_hypers}, using hypers from config.hypers"
        )
        model_hes.covar_module.base_kernel.lengthscale = [[config.hypers["ls"]]]
        # NOTE: GPyTorch outputscale should be set to the SynthFunc alpha squared
        model_hes.covar_module.outputscale = config.hypers["alpha"] ** 2
        model_hes.likelihood.noise_covar.noise = [config.hypers["sigma"]]

        model_hes.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
        model_hes.covar_module.raw_outputscale.requires_grad_(False)
        model_hes.likelihood.noise_covar.raw_noise.requires_grad_(False)
    fit_gpytorch_model(mll_hes)

    optim = torch.optim.Adam([bayes_actions], lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=c.T_max, eta_min=c.eta_min
    )

    patient = c.max_patient
    min_loss = float("inf")
    losses = []
    lrs = []
    for _ in tqdm(range(c.acq_opt_iter)):
        p_yi_xiDi = model_hes.posterior(torch.tanh(bayes_actions))
        batch_yis = sampler(p_yi_xiDi)
        batch_yis = batch_yis.mean(dim=0)
        result = batch_yis.squeeze()
        loss = -result.sum()

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        lrs.append(scheduler.get_last_lr())
        losses.append(loss.cpu().detach().numpy())

        if loss < min_loss:
            min_loss = loss
            patient = c.max_patient
            best_restart = torch.argmax(batch_yis)
            optimal_action = (
                (torch.tanh(bayes_actions))[best_restart, :, :].cpu().detach().numpy()
            )
            eval_metric = result[best_restart].cpu().detach().numpy()
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
    plt.savefig(
        f"{c.save_dir}/{c.algo}/acq_opt_eval_{iteration}.png", bbox_inches="tight"
    )

    # Plot optimal_action in special eval plot here
    plot_topk(
        config = config,
        data = data,
        iteration = iteration,
        next_x = next_x,
        previous_x = previous_x,
        actions = optimal_action,
        eval=True,
    )

    # Return eval_metric and optimal_action (or None)
    return eval_metric, optimal_action


    
if __name__ == "__main__":

    parser = ArgumentParser()
    # Parse args
    parser.add_argument("--app", type=str, default="topk")
    parser.add_argument("--dataset", type=str, default="synthetic")
    parser.add_argument("--algo", type=str, default="hes_mc")
    parser.add_argument("--dist_weight", type=int, default=20.0)
    parser.add_argument("--dist_threshold", type=int, default=2.5)
    parser.add_argument("--exp_id", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    
    # Init params
    parms = Parameters(args)
    
    # Make save dir
    make_save_dir(parms)

    # Set any initial data
    initial_data = None
    if parms.start_iter > 0:
        initial_data = get_init_data(
            path_str=parms.save_dir, start_iter=parms.start_iter, n_init_data=0
        )

    # Init environment
    env = make_env(parms)
    
    # Run hes trial
    run_hes_trial(
        parms=parms,
        env=env,
        initial_data=initial_data,
        plot_function=None,
        eval_function=eval_topk,
        final_eval_function=None,
    )
