import torch


def compute_expectedloss_topk(model, actions, sampler, info, total_cost) -> torch.Tensor:
    r"""
    Evaluate a batch of H-entropy acquisition function for top-K with diversity.
    Note that the model object `model` here is the result of fantasization
    on the intial model. In other words, it is a GP of p (f | D1)
    = p( f | D U {batch_xs, Y}), where Y is sample from posterior p( f | D ).
    This GP represents the inner distribution that we need to compute the expectation
    over.

    See equation 7 in the H-entropy paper.
    For a given function f and action a, the loss function is defined as
        l(f, a) = \sum_i f(a_i) + \sum_{1 \leq i \leq j \leq k} d(a_i, a_j)
    Then the corresponding acquisition function is defined as
        E_{p(y|x,D)} E_{f|D_1} [ l(f, a) + c_t ]
    Value of the acquisition function is approximated using Monte Carlo.
    """
    
    post_pred_dist = [model.posterior(actions[..., k, :]) for k in range(info.n_actions)]
    batch_yis = [sampler(ppd) for ppd in post_pred_dist]
    batch_yis = torch.stack(batch_yis, dim=-2).mean(dim=0).squeeze(-1)
    # >> Tensor[*[n_samples]*i, n_restarts, 1, 1]

    # compute pairwise-distance d(a_i, a_j) for the diversity
    if info.n_actions < 2:
        dist_reward = 0
    
    else:
        actions_dist = torch.cdist(actions.contiguous(), actions.contiguous(), p=1.0)
        # >>> n_samples x n_restarts x K x K

        actions_dist_triu = torch.triu(actions_dist)

        actions_dist_triu[actions_dist_triu > info.dist_threshold] = info.dist_threshold
        dist_reward = actions_dist_triu.sum((-1, -2)) / (info.n_actions * (info.n_actions - 1) / 2.0)  
        # >>> n_samples x n_restarts

    dist_reward = (info.dist_weight * dist_reward)

    # sum over samples from posterior predictive
    E_p = batch_yis.sum(-1) + dist_reward - total_cost
    while len(E_p.shape) > 1:
            E_p = E_p.mean(0)
            
    return E_p


def compute_expectedloss_minmax(model, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on actions.

    Args:
        actions: n_restarts x n_samples x n_actions x action_dim
    """
    assert actions.shape[2] == 2

    # Permute shape of actions to work with model.posterior correctly
    actions = torch.permute(actions, [1, 0, 2, 3])

    posterior = model.posterior(actions)
    # n_fs x n_samples x n_restarts x K x 1
    samples = sampler(posterior)
    val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
    val[:, :, :, 0] = -1 * val[:, :, :, 0]
    val = val.sum(dim=-1)  # n_fs x n_samples x n_restarts
    val = val.mean(dim=0)  # n_samples x n_restarts
    q_hes = val

    return q_hes.mean(dim=0)


def compute_expectedloss_twoval(model, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on actions.

    Args:
        actions: n_restarts x n_samples x n_actions x action_dim
    """
    K = info.n_actions
    assert K == 2

    # Permute shape of actions to work with model.posterior correctly
    actions = torch.permute(actions, [1, 0, 2, 3])

    posterior = model.posterior(actions)
    # n_fs x n_samples x n_restarts x K x 1
    samples = sampler(posterior)
    val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
    val[:, :, :, 0] = -1 * torch.abs(val[:, :, :, 0] - info.val_tuple[0])
    val[:, :, :, 1] = -1 * torch.abs(val[:, :, :, 1] - info.val_tuple[1])
    val = val.sum(dim=-1)  # n_fs x n_samples x n_restarts
    val = val.mean(dim=0)  # n_samples x n_restarts
    q_hes = val

    close = True
    if close:
        actions_dist = torch.cdist(
            actions.contiguous(), actions.contiguous(), p=1.0
        )
        # n_samples x n_restarts x K x K
        actions_dist_triu = torch.triu(actions_dist)
        # n_samples x n_restarts
        dist_reward = -1 * \
            actions_dist_triu.sum((-1, -2)) / (K * (K - 1) / 2.0)
        dist_reward = 2 * dist_reward
        q_hes += dist_reward

    origin = True
    if origin:
        dist_origin_reward = -20 * \
            torch.linalg.norm(actions[:, :, 0, :], dim=-1)
        dist_origin_reward += -5 * \
            torch.linalg.norm(actions[:, :, 1, :], dim=-1)
        q_hes += dist_origin_reward

    return q_hes.mean(dim=0)


def compute_expectedloss_mvs(model, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on actions.

    Args:
        actions: n_restarts x n_samples x n_actions x action_dim
    """

    K = info.n_actions

    # Permute shape of actions to work with model.posterior correctly
    actions = torch.permute(actions, [1, 0, 2, 3])

    posterior = model.posterior(actions)
    # n_fs x n_samples x n_restarts x K x 1
    samples = sampler(posterior)
    val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
    for idx in range(K):
        # val[:, :, :, idx] = -1 * torch.abs(val[:, :, :, idx] - info.val_tuple[idx])**2
        val[:, :, :, idx] = -1 * \
            torch.abs(val[:, :, :, idx] - info.val_tuple[idx])

    val = val.sum(dim=-1)  # n_fs x n_samples x n_restarts
    val = val.mean(dim=0)  # n_samples x n_restarts
    q_hes = val

    close = False
    if close:
        actions_dist = torch.cdist(
            actions.contiguous(), actions.contiguous(), p=1.0
        )
        # n_samples x n_restarts x K x K
        actions_dist_triu = torch.triu(actions_dist)
        # n_samples x n_restarts
        dist_reward = -1 * \
            actions_dist_triu.sum((-1, -2)) / (K * (K - 1) / 2.0)
        dist_reward = 1 * dist_reward
        q_hes += dist_reward

    origin = False
    if origin:
        dist_origin_reward = -20 * \
            torch.linalg.norm(actions[:, :, 0, :], dim=-1)
        dist_origin_reward += -5 * \
            torch.linalg.norm(actions[:, :, 1, :], dim=-1)
        q_hes += dist_origin_reward

    chain = True
    if chain:
        for idx in range(1, K):
            link_dist = actions[:, :, idx, :] - actions[:, :, idx - 1, :]
            # link_dist_reward = -0.01 * torch.linalg.norm(link_dist, dim=-1)
            link_dist_reward = -0.1 * torch.linalg.norm(link_dist, dim=-1)
            q_hes += link_dist_reward

    return q_hes.mean(dim=0)


def compute_expectedloss_levelset(model, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on Actions.

    Args:
        support_points: n_actions x data_dim
        actions: n_restarts x n_samples x n_actions x action_dim
        where n_actions is the support size.
    """
    assert actions.shape[3] == 1

    # n_samples x n_restarts x n_actions
    actions = actions.squeeze(-1).permute([1, 0, 2])
    posterior = model.posterior(info.support_points)
    # n_fs x n_samples x n_restarts x n_actions x 1
    samples = sampler(posterior)
    val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x n_actions
    val = val.mean(dim=0)  # n_samples x n_restarts x n_actions
    q_hes = ((val - info.levelset_threshold) * actions).sum(
        dim=-1
    )  # n_samples x n_restarts
    return q_hes.mean(dim=0)


def compute_expectedloss_multilevelset(model, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on Actions.

    Args:
        support_points: n_actions x data_dim
        actions: n_restarts x n_samples x n_actions x action_dim
        where n_actions is the support size.
    """
    assert actions.shape[3] == len(info.levelset_thresholds)
    # n_samples x n_restarts x n_actions x n_levelset
    actions = actions.permute([1, 0, 2, 3])
    posterior = model.posterior(info.support_points)
    # n_fs x n_samples x n_restarts x n_actions x 1
    samples = sampler(posterior)
    val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x n_actions
    val = val.mean(dim=0)  # n_samples x n_restarts x n_actions
    q_hes = 0
    for i, threshold in enumerate(info.levelset_thresholds):
        # n_samples x n_restarts
        q_hes += ((val - threshold) * actions[:, :, :, i]).sum(dim=-1)
    return q_hes.mean(dim=0)


def compute_expectedloss_expf(model, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on actions.
    Args:
        actions: n_restarts x n_samples x n_actions x action_dim
    """

    K = info.config.n_actions

    # Permute shape of actions to work with model.posterior correctly
    actions = torch.permute(actions, [1, 0, 2, 3])

    # Draw samples from Normal distribution
    # --- Draw standard normal samples (of a certain shape)
    # std_normal = torch.normal(# TODO)
    # --- Transform, using actions, to get to correct means/stds/weights (encoded in actions)
    # TODO
    # --- Take function evals with model.posterior(samples)
    # TODO
    # --- Compute average of these function evals
    # TODO

    posterior = model.posterior(actions)
    # n_fs x n_samples x n_restarts x K x 1
    samples = sampler(posterior)
    val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
    val = val.sum(dim=-1)  # n_fs x n_samples x n_restarts
    val = val.mean(dim=0)  # n_samples x n_restarts

    actions_dist = torch.cdist(
        actions.contiguous(), actions.contiguous(), p=1.0)
    # n_samples x n_restarts x K x K
    actions_dist_triu = torch.triu(actions_dist)
    actions_dist_triu[
        actions_dist_triu > info.dist_threshold
    ] = info.dist_threshold
    dist_reward = actions_dist_triu.sum((-1, -2)) / (
        K * (K - 1) / 2.0
    )  # n_samples x n_restarts
    dist_reward = info.dist_weight * dist_reward

    q_hes = val + dist_reward
    q_hes = q_hes.squeeze()
    return q_hes.mean(dim=0)


def compute_expectedloss_pbest(model, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on actions.

    Args:
        actions: n_restarts x n_samples x n_actions x action_dim
    """

    K = info.config.n_actions

    # Permute shape of actions to work with model.posterior correctly
    actions = torch.permute(actions, [1, 0, 2, 3])

    samples_rand_samp = sampler(info.posterior_rand_samp)
    maxes = torch.amax(samples_rand_samp, dim=(3, 4))

    posterior = model.posterior(actions)
    # out shape: n_fs x n_samples x n_restarts x K x 1
    samples = sampler(posterior)

    # out shape: n_fs x n_samples x n_restarts
    val = -1 * (maxes - samples.squeeze())
    val[val > -0.2] = -0.2
    val = val.mean(dim=0)  # out shape: n_samples x n_restarts

    q_hes = val
    q_hes = q_hes.squeeze()
    return q_hes.mean(dim=0)


def compute_expectedloss_bestofk(model, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on actions.

    Args:
        actions: n_restarts x n_samples x n_actions x action_dim
    """

    K = info.config.n_actions

    # Permute shape of actions to work with model.posterior correctly
    actions = torch.permute(actions, [1, 0, 2, 3])

    posterior = model.posterior(actions)
    # n_fs x n_samples x n_restarts x K x 1
    samples = sampler(posterior)
    val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
    val = val.amax(dim=-1)  # n_fs x n_samples x n_restarts
    val = val.mean(dim=0)  # n_samples x n_restarts

    q_hes = val
    q_hes = q_hes.squeeze()
    return q_hes.mean(dim=0)
