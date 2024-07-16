import os
import torch
import random
import argparse
import itertools
import numpy as np
from tueplots import bundles
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.linalg import solve_triangular
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
from botorch.sampling.normal import SobolQMCNormalSampler

from acqfs import qBOAcqf

# from synthetic_functions.alpine import AlpineN1
# from synthetic_functions.syngp import SynGP
from env_wrapper import EnvWrapper

plt.rcParams.update(bundles.iclr2023())


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
    f.noise_std = noise_std * (f.range_y[1] - f.range_y[0]) / 100

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
        prev_X = buffer["x"][iteration:iteration + 1].expand(parms.n_restarts, -1)
        if embedder is not None:
            # Discretize: Continuous -> Discrete
            prev_X = embedder.decode(prev_X)
            prev_X = torch.nn.functional.one_hot(
                prev_X, num_classes=parms.num_categories
            ).to(dtype=parms.torch_dtype)
            # >>> n_restarts x x_dim x n_categories

        prev_y = (
            buffer["y"][iteration:iteration + 1]
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
    regret = 1 - u_posterior

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


def unif_random_sample_domain(domain, n=1):
    """Draws a sample uniformly at random from domain (a list of tuple bounds)."""
    list_of_arr_per_dim = [np.random.uniform(dom[0], dom[1], n) for dom in domain]
    list_of_list_per_sample = [list(la) for la in np.array(list_of_arr_per_dim).T]
    return list_of_list_per_sample


def kern_exp_quad_ard(xmat1, xmat2, ls, alpha):
    """
    Exponentiated quadratic kernel function with
    dimensionwise lengthscales if ls is an ndarray.
    """
    xmat1 = np.expand_dims(xmat1, axis=1)
    xmat2 = np.expand_dims(xmat2, axis=0)
    diff = xmat1 - xmat2
    diff /= ls
    norm = np.sum(diff**2, axis=-1) / 2.0
    kern = alpha**2 * np.exp(-norm)
    return kern


def kern_exp_quad_ard_sklearn(xmat1, xmat2, ls, alpha):
    """
    Exponentiated quadratic kernel function with dimensionwise lengthscales if ls is an
    ndarray, based on scikit-learn implementation.
    """
    dists = cdist(xmat1 / ls, xmat2 / ls, metric="sqeuclidean")
    exp_neg_norm = np.exp(-0.5 * dists)
    return alpha**2 * exp_neg_norm


def kern_exp_quad_ard_per(xmat1, xmat2, ls, alpha, pdims, period=2):
    """
    Exponentiated quadratic kernel function with
    - dimensionwise lengthscales if ls is an ndarray
    - periodic dimensions denoted by pdims. We assume that the period
    is 2.
    """
    xmat1 = np.expand_dims(xmat1, axis=1)
    xmat2 = np.expand_dims(xmat2, axis=0)
    diff = xmat1 - xmat2
    diff[..., pdims] = np.sin((np.pi * diff[..., pdims] / period) % (2 * np.pi))
    # diff[..., pdims] = np.cos( (np.pi/2) + (np.pi * diff[..., pdims] / period) )
    diff /= ls
    norm = np.sum(diff**2, axis=-1) / 2.0
    kern = alpha**2 * np.exp(-norm)

    return kern


def kern_exp_quad_noard(xmat1, xmat2, ls, alpha):
    """
    Exponentiated quadratic kernel function (aka squared exponential kernel aka
    RBF kernel).
    """
    kern = alpha**2 * kern_exp_quad_noard_noscale(xmat1, xmat2, ls)
    return kern


def kern_exp_quad_noard_noscale(xmat1, xmat2, ls):
    """
    Exponentiated quadratic kernel function (aka squared exponential kernel aka
    RBF kernel), without scale parameter.
    """
    distmat = squared_euc_distmat(xmat1, xmat2)
    norm = distmat / (2 * ls**2)
    exp_neg_norm = np.exp(-norm)
    return exp_neg_norm


def squared_euc_distmat(xmat1, xmat2, coef=1.0):
    """
    Distance matrix of squared euclidean distance (multiplied by coef) between
    points in xmat1 and xmat2.
    """
    return coef * cdist(xmat1, xmat2, "sqeuclidean")


def kern_distmat(xmat1, xmat2, ls, alpha, distfn):
    """
    Kernel for a given distmat, via passed in distfn (which is assumed to be fn
    of xmat1 and xmat2 only).
    """
    distmat = distfn(xmat1, xmat2)
    kernmat = alpha**2 * np.exp(-distmat / (2 * ls**2))
    return kernmat


def kern_simple_list(xlist1, xlist2, ls, alpha, base_dist=5.0):
    """
    Kernel for two lists containing elements that can be compared for equality.
    K(a,b) = 1 + base_dist if a and b are equal and K(a,b) = base_dist otherwise.
    """
    distmat = simple_list_distmat(xlist1, xlist2)
    distmat = distmat + base_dist
    kernmat = alpha**2 * np.exp(-distmat / (2 * ls**2))
    return kernmat


def simple_list_distmat(xlist1, xlist2, weight=1.0, additive=False):
    """
    Return distance matrix containing zeros when xlist1[i] == xlist2[j] and 0 otherwise.
    """
    prod_list = list(itertools.product(xlist1, xlist2))
    len1 = len(xlist1)
    len2 = len(xlist2)
    try:
        binary_mat = np.array([x[0] != x[1] for x in prod_list]).astype(int)
    except RuntimeWarning:
        # For cases where comparison returns iterable of bools
        binary_mat = np.array([all(x[0] != x[1]) for x in prod_list]).astype(int)

    binary_mat = binary_mat.reshape(len1, len2)

    if additive:
        distmat = weight + binary_mat
    else:
        distmat = weight * binary_mat

    return distmat


def get_product_kernel(kernel_list, additive=False):
    """Given a list of kernel functions, return product kernel."""

    def product_kernel(x1, x2, ls, alpha):
        """Kernel returning elementwise-product of kernel matrices from kernel_list."""
        mat_prod = kernel_list[0](x1, x2, ls, 1.0)
        for kernel in kernel_list[1:]:
            if additive:
                mat_prod = mat_prod + kernel(x1, x2, ls, 1.0)
            else:
                mat_prod = mat_prod * kernel(x1, x2, ls, 1.0)
        mat_prod = alpha**2 * mat_prod
        return mat_prod

    return product_kernel


def get_cholesky_decomp(k11_nonoise, sigma, psd_str):
    """Return cholesky decomposition."""
    if psd_str == "try_first":
        k11 = k11_nonoise + sigma**2 * np.eye(k11_nonoise.shape[0])
        try:
            return stable_cholesky(k11, False)
        except np.linalg.linalg.LinAlgError:
            return get_cholesky_decomp(k11_nonoise, sigma, "project_first")
    elif psd_str == "project_first":
        k11_nonoise = project_symmetric_to_psd_cone(k11_nonoise)
        return get_cholesky_decomp(k11_nonoise, sigma, "is_psd")
    elif psd_str == "is_psd":
        k11 = k11_nonoise + sigma**2 * np.eye(k11_nonoise.shape[0])
        return stable_cholesky(k11)


def stable_cholesky(mmat, make_psd=True, verbose=False):
    """Return a 'stable' cholesky decomposition of mmat."""
    if mmat.size == 0:
        return mmat
    try:
        lmat = np.linalg.cholesky(mmat)
    except np.linalg.linalg.LinAlgError as e:
        if not make_psd:
            raise e
        diag_noise_power = -11
        max_mmat = np.diag(mmat).max()
        diag_noise = np.diag(mmat).max() * 1e-11
        break_loop = False
        while not break_loop:
            try:
                lmat = np.linalg.cholesky(
                    mmat + ((10**diag_noise_power) * max_mmat) * np.eye(mmat.shape[0])
                )
                break_loop = True
            except np.linalg.linalg.LinAlgError:
                if diag_noise_power > -9:
                    if verbose:
                        print(
                            "\t*stable_cholesky failed with "
                            "diag_noise_power=%d." % (diag_noise_power)
                        )
                diag_noise_power += 1
            if diag_noise_power >= 5:
                print("\t*stable_cholesky failed: added diag noise = %e" % (diag_noise))
    return lmat


def project_symmetric_to_psd_cone(mmat, is_symmetric=True, epsilon=0):
    """Project symmetric matrix mmat to the PSD cone."""
    if is_symmetric:
        try:
            eigvals, eigvecs = np.linalg.eigh(mmat)
        except np.linalg.LinAlgError:
            print("\tLinAlgError encountered with np.eigh. Defaulting to eig.")
            eigvals, eigvecs = np.linalg.eig(mmat)
            eigvals = np.real(eigvals)
            eigvecs = np.real(eigvecs)
    else:
        eigvals, eigvecs = np.linalg.eig(mmat)
    clipped_eigvals = np.clip(eigvals, epsilon, np.inf)
    return (eigvecs * clipped_eigvals).dot(eigvecs.T)


def solve_lower_triangular(amat, b):
    """Solves amat*x=b when amat is lower triangular."""
    return solve_triangular_base(amat, b, lower=True)


def solve_upper_triangular(amat, b):
    """Solves amat*x=b when amat is upper triangular."""
    return solve_triangular_base(amat, b, lower=False)


def solve_triangular_base(amat, b, lower):
    """Solves amat*x=b when amat is a triangular matrix."""
    if amat.size == 0 and b.shape[0] == 0:
        return np.zeros((b.shape))
    else:
        return solve_triangular(amat, b, lower=lower)


def sample_mvn(mu, covmat, nsamp):
    """
    Sample from multivariate normal distribution with mean mu and covariance
    matrix covmat.
    """
    mu = mu.reshape(-1)
    ndim = len(mu)
    lmat = stable_cholesky(covmat)
    umat = np.random.normal(size=(ndim, nsamp))
    return lmat.dot(umat).T + mu


def gp_post(x_train, y_train, x_pred, ls, alpha, sigma, kernel, full_cov=True):
    """Compute parameters of GP posterior"""
    k11_nonoise = kernel(x_train, x_train, ls, alpha)
    lmat = get_cholesky_decomp(k11_nonoise, sigma, "try_first")
    smat = solve_upper_triangular(lmat.T, solve_lower_triangular(lmat, y_train))
    k21 = kernel(x_pred, x_train, ls, alpha)
    mu2 = k21.dot(smat)
    k22 = kernel(x_pred, x_pred, ls, alpha)
    vmat = solve_lower_triangular(lmat, k21.T)
    k2 = k22 - vmat.T.dot(vmat)
    if full_cov is False:
        k2 = np.sqrt(np.diag(k2))
    return mu2, k2
