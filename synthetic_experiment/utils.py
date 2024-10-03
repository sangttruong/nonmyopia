import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from acqfs import qBOAcqf
from botorch.sampling.normal import SobolQMCNormalSampler
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
from synthetic_functions.alpine import AlpineN1
from synthetic_functions.env_wrapper import EnvWrapper
from synthetic_functions.syngp import SynGP
from tueplots import bundles

plt.rcParams.update(bundles.neurips2024())


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(name, x_dim, bounds, noise_std=0.0):
    r"""Make environment."""
    if name == "Ackley":
        f_ = Ackley(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "Ackley4D":
        f_ = Ackley(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "Alpine":
        f_ = AlpineN1(dim=x_dim, noise_std=noise_std)
    elif name == "Beale":
        f_ = Beale(negate=True, noise_std=noise_std)
    elif name == "Branin":
        f_ = Branin(negate=True, noise_std=noise_std)
    elif name == "Cosine8":
        f_ = Cosine8(noise_std=noise_std)
    elif name == "EggHolder":
        f_ = EggHolder(negate=True, noise_std=noise_std)
    elif name == "Griewank":
        f_ = Griewank(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "Hartmann":
        f_ = Hartmann(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "HolderTable":
        f_ = HolderTable(negate=True, noise_std=noise_std)
    elif name == "Levy":
        f_ = Levy(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "Powell":
        f_ = Powell(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "Rosenbrock":
        f_ = Rosenbrock(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "SixHumpCamel":
        f_ = SixHumpCamel(negate=True, noise_std=noise_std)
    elif name == "StyblinskiTang":
        f_ = StyblinskiTang(dim=x_dim, negate=True, noise_std=noise_std)
    elif name == "SynGP":
        f_ = SynGP(dim=x_dim, noise_std=noise_std)
    else:
        raise NotImplementedError

    # Set env bound
    f_.bounds[0, :] = bounds[..., 0]
    f_.bounds[1, :] = bounds[..., 1]

    # Wrapper for normalizing output
    f = EnvWrapper(name, f_)

    return f


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def make_save_dir(config):
    r"""Create save directory without overwriting directories."""
    init_dir_path = Path(config.save_dir)
    dir_path = Path(str(init_dir_path))

    if not os.path.exists(os.path.join(config.save_dir, "buffer.pt")):
        config.cont = False

    if not config.cont and not os.path.exists(config.save_dir):
        dir_path.mkdir(parents=True, exist_ok=False)
    elif not config.cont and os.path.exists(config.save_dir):
        config.save_dir = str(dir_path)
    elif config.cont and not os.path.exists(config.save_dir):
        print(
            f"WARNING: save_dir={config.save_dir} does not exist. Rerun from scratch."
        )
        dir_path.mkdir(parents=True, exist_ok=False)

    print(f"Created save_dir: {config.save_dir}")

    # Save config to save_dir as parameters.json
    config_path = dir_path / "parameters.json"
    with open(str(config_path), "w", encoding="utf-8") as file_handle:
        config_dict = str(config)
        file_handle.write(config_dict)


def eval_func(env, model, parms, buffer, iteration, embedder=None, *args, **kwargs):
    # Quality of the best decision from the current posterior distribution ###
    cost_fn = parms.cost_function_class(**parms.cost_func_hypers)
    previous_cost = (
        buffer["cost"][: iteration + 1].sum()
        if iteration + 1 > parms.n_initial_points
        else 0.0
    )

    u_observed = torch.max(buffer["y"][: iteration + 1]).item()

    ######################################################################

    if parms.algo.startswith("HES"):
        # Initialize A consistently across fantasies
        A = buffer["x"][iteration].clone().repeat(parms.n_restarts, parms.n_actions, 1)
        A = A + torch.randn_like(A) * 0.01
        if embedder is not None:
            A = embedder.decode(A)
            A = torch.nn.functional.one_hot(A, num_classes=parms.num_categories).to(
                parms.torch_dtype
            )
        A.requires_grad = True

        # Initialize optimizer
        optimizer = torch.optim.AdamW([A], lr=parms.acq_opt_lr)
        loss_fn = parms.loss_function_class(**parms.loss_func_hypers)

        for i in range(1000):
            if embedder is not None:
                actions = embedder.encode(A)
            else:
                actions = A
            ppd = model(actions)
            y_A = ppd.rsample()

            losses = loss_fn(A=actions, Y=y_A) + cost_fn(
                prev_X=buffer["x"][iteration].expand_as(actions),
                current_X=actions,
                previous_cost=previous_cost,
            )
            # >>> n_fantasy x batch_size

            loss = losses.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (i + 1) % 200 == 0:
                print(f"Eval optim round: {i+1}, Loss: {loss.item():.2f}")

        aidx = losses.squeeze(-1).argmin()

    else:
        # Construct acqf
        sampler = SobolQMCNormalSampler(sample_shape=1, seed=0, resample=False)
        nf_design_pts = [1]

        acqf = qBOAcqf(
            name=parms.algo,
            model=model,
            lookahead_steps=0 if parms.algo_lookahead_steps == 0 else 1,
            n_actions=parms.n_actions,
            n_fantasy_at_design_pts=nf_design_pts,
            loss_function_class=parms.loss_function_class,
            loss_func_hypers=parms.loss_func_hypers,
            cost_function_class=parms.cost_function_class,
            cost_func_hypers=parms.cost_func_hypers,
            sampler=sampler,
            best_f=buffer["y"][: iteration + 1].max(),
        )

        maps = []
        if parms.algo_lookahead_steps > 0:
            x = buffer["x"][iteration].clone().repeat(parms.n_restarts, 1)
            if embedder is not None:
                x = embedder.decode(x)
                x = torch.nn.functional.one_hot(x, num_classes=parms.num_categories).to(
                    parms.torch_dtype
                )
            maps.append(x)

        A = buffer["x"][iteration].clone().repeat(parms.n_restarts * parms.n_actions, 1)
        A = A + torch.randn_like(A) * 0.01
        if embedder is not None:
            A = embedder.decode(A)
            A = torch.nn.functional.one_hot(A, num_classes=parms.num_categories).to(
                parms.torch_dtype
            )
        A.requires_grad = True
        maps.append(A)

        # Initialize optimizer
        optimizer = torch.optim.AdamW([A], lr=parms.acq_opt_lr)

        # Get prevX, prevY
        prev_X = buffer["x"][iteration : iteration + 1].expand(parms.n_restarts, -1)
        if embedder is not None:
            # Discretize: Continuous -> Discrete
            prev_X = embedder.decode(prev_X)
            prev_X = torch.nn.functional.one_hot(
                prev_X, num_classes=parms.num_categories
            ).to(dtype=parms.torch_dtype)
            # >>> n_restarts x x_dim x n_categories

        prev_y = (
            buffer["y"][iteration : iteration + 1]
            .expand(parms.n_restarts, -1)
            .to(dtype=parms.torch_dtype)
        )

        for i in range(1000):
            return_dict = acqf.forward(
                prev_X=prev_X,
                prev_y=prev_y,
                prev_hid_state=None,
                maps=maps,
                embedder=embedder,
                prev_cost=previous_cost,
            )

            losses = return_dict["acqf_loss"] + return_dict["acqf_cost"]
            # >>> n_fantasy_at_design_pts x batch_size

            loss = losses.mean()
            grads = torch.autograd.grad(loss, [A], allow_unused=True)
            for param, grad in zip([A], grads):
                param.grad = grad
            optimizer.step()
            optimizer.zero_grad()

            if (i + 1) % 200 == 0:
                print(f"Eval optim round: {i}, Loss: {loss.item():.2f}")

        aidx = losses.squeeze(-1).argmin()
        if embedder is not None:
            A = A.reshape(
                parms.n_restarts, parms.n_actions, parms.x_dim, parms.num_categories
            )
        else:
            A = A.reshape(parms.n_restarts, parms.n_actions, parms.x_dim)

    if embedder is not None:
        A_chosen = embedder.encode(A[aidx]).detach()
    else:
        A_chosen = A[aidx].detach()
    u_posterior = env(A_chosen).item()

    ######################################################################
    regret = 3 - u_posterior

    return (u_observed, u_posterior, regret), A_chosen.cpu()


def draw_loss_and_cost(save_dir, losses, costs, iteration):
    plt.plot(losses, label="Loss value")
    plt.plot(costs, label="Cost value")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Loss and cost value at iteration {iteration}")
    plt.legend()
    plt.savefig(f"{save_dir}/losses_and_costs_{iteration}.pdf")
    plt.close()


def generate_random_points_batch(inputs, radius, n):
    # Get the shape of the inputs (excluding the last dimension)
    batch_shape = inputs.shape[:-1]
    dim = inputs.shape[-1]  # Dimensionality of the points

    # Expand O to the shape [n, batch0, batch1, ..., dim] for broadcasting
    O_expanded = inputs.unsqueeze(0).expand(n, *batch_shape, dim)

    # Generate random directions for each point
    directions = torch.randn(n, *batch_shape, dim).to(inputs)  # Generate random normals
    directions = directions / torch.norm(
        directions, dim=-1, keepdim=True
    )  # Normalize to unit length

    # Generate random radii between 0 and radius for each point
    radii = (
        torch.rand(n, *batch_shape).pow(1 / dim).unsqueeze(-1).to(inputs) * radius
    )  # Scale radius

    # Calculate the random points by scaling the direction by the radius and adding to O
    random_points = O_expanded + radii * directions

    return random_points


def generate_random_rotation_matrix(D):
    # Generate a random matrix
    random_matrix = torch.randn(D, D)

    # Perform QR decomposition to get an orthogonal matrix
    Q, _ = torch.linalg.qr(random_matrix)

    # Ensure the determinant is 1 (proper rotation)
    if torch.det(Q) < 0:
        # Flip the sign of the random column if the determinant is -1
        Q[:, random.randint(0, D - 1)] *= -1

    return Q


def rotate_points(inputs, R, p):
    # Translate inputs so that p becomes the origin
    translated_inputs = inputs - p

    # Apply the rotation matrix to the translated inputs
    rotated_translated = torch.matmul(
        translated_inputs, R.transpose(-1, -2).to(translated_inputs)
    )

    # Translate points back to the original position relative to p
    rotated_points = rotated_translated + p

    return rotated_points
