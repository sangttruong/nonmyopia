from botorch.test_functions.synthetic import (
    Ackley,  # XD Ackley function - Minimum
    Beale,  # 2D Beale function - Minimum
    Branin,  # 2D Branin function - Minimum
    Cosine8,  # 8D cosine function - Maximum,
    EggHolder,  # 2D EggHolder function - Minimum
    Griewank,  # XD Griewank function - Minimum
    Hartmann,  # 6D Hartmann function - Minimum
    HolderTable,  # 2D HolderTable function - Minimum
    Levy,  # XD Levy function - Minimum
    Powell,  # 4D Powell function - Minimum
    Rosenbrock,  # XD Rosenbrock function - Minimum
    SixHumpCamel,  # 2D SixHumpCamel function - Minimum
    StyblinskiTang,  # XD StyblinskiTang function - Minimum
)
from _0_main import Parameters, make_env
from _4_bo_acqf import qBOAcqf
from _7_utils import set_seed
from _12_alpine import AlpineN1
from _13_embedder import DiscreteEmbbeder
from _15_syngp import SynGP
from _16_env_wrapper import EnvWrapper
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from tqdm import tqdm
import os
import argparse
from tensordict import TensorDict
from pathlib import Path
from argparse import ArgumentParser
import wandb

# Compute metrics
def compute_metrics(
    env,
    parms,
    buffer,
    embedder,
    device,
    WM_state_dicts=None,
):
    metrics = []
    cost_fn = parms.cost_function_class(**parms.cost_func_hypers)
    previous_cost = 0
    
    for step in tqdm(range(parms.n_initial_points-1, parms.algo_n_iterations)):
        u_observed = torch.max(buffer["y"][:step+1]).item()

        ######################################################################
        WM = SingleTaskGP(
            buffer["x"][:step+1],
            buffer["y"][:step+1],
            # input_transform=Normalize(
            #     d=parms.x_dim, bounds=parms.bounds.T),
            # outcome_transform=Standardize(1),
        ).to(device)
        
        if step == parms.algo_n_iterations - 1:
            # Fit GP
            mll = ExactMarginalLogLikelihood(WM.likelihood, WM)
            fit_gpytorch_model(mll)
        else:
            WM.load_state_dict(WM_state_dicts[step - parms.n_initial_points + 1])

        # Set WM to eval mode
        WM.eval()
        
        if parms.algo.startswith("HES"):
            # Initialize A consistently across fantasies
            A = torch.empty(
                [parms.n_restarts, parms.n_actions, parms.x_dim],
                device=device
            )
            A = buffer["x"][step].clone().repeat(
                parms.n_restarts, parms.n_actions, 1)
            A = A + torch.randn_like(A) * 0.01
            if embedder is not None:
                A = embedder.decode(A)
                A = torch.nn.functional.one_hot(A, num_classes=parms.num_categories).float()
            A.requires_grad = True
    
            # Initialize optimizer
            optimizer = torch.optim.AdamW([A], lr=parms.acq_opt_lr)
            loss_fn = parms.loss_function_class(**parms.loss_func_hypers)
    
            for i in range(1000):
                if embedder is not None:
                    actions = embedder.encode(A)
                else:
                    actions = A
                ppd = WM(actions)
                y_A = ppd.rsample()
    
                losses = loss_fn(A=actions, Y=y_A) + cost_fn(
                    prev_X=buffer["x"][step].expand_as(actions), current_X=actions, previous_cost=previous_cost
                )
                # >>> n_fantasy x batch_size
    
                loss = losses.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
                if (i+1) % 200 == 0:
                    print(f"Eval optim round: {i+1}, Loss: {loss.item():.2f}")
    
            aidx = losses.squeeze(-1).argmin()
            
        else:
            # Construct acqf
            sampler = SobolQMCNormalSampler(
                sample_shape=1, seed=0, resample=False
            )
            nf_design_pts = [1]
            
            acqf = qBOAcqf(
                name=parms.algo,
                model=WM, 
                lookahead_steps=0 if parms.algo_lookahead_steps == 0 else 1,
                n_actions=parms.n_actions,
                n_fantasy_at_design_pts=nf_design_pts,
                loss_function_class=parms.loss_function_class,
                loss_func_hypers=parms.loss_func_hypers,
                cost_function_class=parms.cost_function_class,
                cost_func_hypers=parms.cost_func_hypers,
                sampler=sampler,
                best_f=buffer["y"][:step+1].max(),
            )

            maps = []
            if parms.algo_lookahead_steps > 0:
                x = buffer["x"][step].clone().repeat(
                    parms.n_restarts, 1)
                if embedder is not None:
                    x = embedder.decode(x)
                    x = torch.nn.functional.one_hot(x, num_classes=parms.num_categories).float()
                maps.append(x)

            A = buffer["x"][step].clone().repeat(
                parms.n_restarts * parms.n_actions, 1)
            A = A + torch.randn_like(A) * 0.01
            if embedder is not None:
                A = embedder.decode(A)
                A = torch.nn.functional.one_hot(A, num_classes=parms.num_categories).float()
            A.requires_grad = True
            maps.append(A)
            
            # Initialize optimizer
            optimizer = torch.optim.AdamW([A], lr=parms.acq_opt_lr)

            # Get prevX, prevY
            prev_X = buffer["x"][step: step+1].expand(
                parms.n_restarts, -1
            )
            if embedder is not None:
                # Discretize: Continuous -> Discrete
                prev_X = embedder.decode(prev_X)
                prev_X = torch.nn.functional.one_hot(
                    prev_X, num_classes=parms.num_categories
                ).to(dtype=parms.torch_dtype)
                # >>> n_restarts x x_dim x n_categories
    
            prev_y = buffer["y"][step: step+1].expand(
                parms.n_restarts, -1
            )

            for i in range(1000):
                return_dict = acqf.forward(
                    prev_X=prev_X,
                    prev_y=prev_y,
                    prev_hid_state=None,
                    maps=maps,
                    embedder=embedder,
                    prev_cost=previous_cost
                )
                
                losses = (return_dict["acqf_loss"] + return_dict["acqf_cost"])
                # >>> n_fantasy_at_design_pts x batch_size

                loss = losses.mean()
                grads = torch.autograd.grad(
                    loss, [A], allow_unused=True)
                for param, grad in zip([A], grads):
                    param.grad = grad
                optimizer.step()
                optimizer.zero_grad()

                if (i+1) % 200 == 0:
                    print(f"Eval optim round: {i}, Loss: {loss.item():.2f}")

            aidx = losses.squeeze(-1).argmin()
            if embedder is not None:
                A = A.reshape(parms.n_restarts, parms.n_actions, parms.x_dim, parms.num_categories)
            else:
                A = A.reshape(parms.n_restarts, parms.n_actions, parms.x_dim)
        
        if embedder is not None:
            A_chosen = embedder.encode(A[aidx]).detach()
        else:
            A_chosen = A[aidx].detach()
        u_posterior = env(A_chosen).item()

        ######################################################################
        regret = 1 - u_posterior
        if step >= parms.n_initial_points:
            regret += metrics[-1][-1] # Cummulative regret

        previous_cost = cost_fn(
            prev_X=buffer["x"][step:step+1], current_X=buffer["x"][step+1:step+2], previous_cost=previous_cost
        )
        metrics.append([u_observed, u_posterior, regret])
        # print({"u_observed": u_observed, "u_posterior": u_posterior, "c_regret": regret})
        wandb.log({"u_observed": u_observed, "u_posterior": u_posterior, "c_regret": regret})

    return np.array(metrics)


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
if __name__ == '__main__':
    wandb.init(project="nonmyopia-metrics")

    # Parse args
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--task", type=str, default="topk")
    parser.add_argument("--env_name", type=str, default="SynGP")
    parser.add_argument("--env_noise", type=float, default=0.0)
    parser.add_argument("--env_discretized", type=str2bool, default=False)
    parser.add_argument("--algo", type=str, default="HES-TS-AM-20")
    parser.add_argument("--cost_fn", type=str, default="euclidean")
    parser.add_argument("--plot", type=str2bool, default=False)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--cont", type=str2bool, default=True)
    args = parser.parse_args()
    
    set_seed(args.seed)

    local_parms = Parameters(args)

    seed = args.seed
    env_name = args.env_name
    env_noise = args.env_noise
    env_discretized = args.env_discretized
    algo = args.algo
    cost_fn = args.cost_fn
    
    # Init environment
    env = make_env(
        name=local_parms.env_name,
        x_dim=local_parms.x_dim,
        bounds=local_parms.bounds,
        noise_std=local_parms.env_noise,
    )
    env = env.to(
        dtype=local_parms.torch_dtype,
        device=local_parms.device,
    )
    
    print(f"Computing metrics for {algo}, {cost_fn}, seed{seed}")
    if False: # os.path.exists(f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy"):
        metrics = np.load(
            f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy")
    else:
        buffer_file=f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/buffer.pt"
        if not os.path.exists(buffer_file):
            raise RuntimeError(f"File {buffer_file} does not exist")
        buffer=torch.load(buffer_file, map_location=device)
        
        WM_state_dicts=[]
        for step in range(local_parms.n_initial_points, local_parms.algo_n_iterations):
            WM_state_dict_file=f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/world_model_{step}.pt"
            if not os.path.exists(WM_state_dict_file):
                raise RuntimeError(f"File {WM_state_dict_file} does not exist")
                
            WM_state_dicts.append(torch.load(WM_state_dict_file, map_location=device))

        if len(WM_state_dicts) != (local_parms.algo_n_iterations - local_parms.n_initial_points):
            raise RuntimeError

        if local_parms.env_discretized:
            embedder = DiscreteEmbbeder(
                num_categories=local_parms.num_categories,
                bounds=local_parms.bounds,
            ).to(device=local_parms.device, dtype=local_parms.torch_dtype)
        else:
            embedder = None
    
        metrics=compute_metrics(
            env,
            local_parms,
            buffer,
            embedder,
            device,
            WM_state_dicts=WM_state_dicts,
        )
        with open(f"results/{env_name}_{env_noise}{'_discretized' if env_discretized else ''}/{algo}_{cost_fn}_seed{seed}/metrics.npy", "wb") as f:
            np.save(f, metrics)
            
    wandb.finish()