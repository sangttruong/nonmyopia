import torch


def compute_expectedloss_topk(post_dist, actions, sampler, info) -> torch.Tensor:
    r"""
    Evaluate a batch of H-entropy acquisition function for top-K with diversity.
    Note that the model object `self.model` here is the result of fantasization
    on the intial model. In other words, it is a GP of p (f | D1)
    = p( f | D U {batch_xs, Y}), where Y is sample from posterior p( f | D ).
    This GP represents the inner distribution that we need to compute the expectation
    over.

    See equation 7 in the H-entropy paper.
    For a given function f and action a, the loss function is defined as
        l(f, a) = \sum_i f(a_i) + \sum_{1 \leq i \leq j \leq k} d(a_i, a_j)
    Then the corresponding acquisition function is defined as
        E_{p(y|x,D)} E_{f|D_1} [ l(f, a) ]
    Value of the acquisition function is approximated using Monte Carlo.
    """

    post_pred_dist = post_dist.posterior(action) 
    sample = self.sampler(post_pred_dist)

    batch_yis = batch_yis.mean(dim=0)
    # >> Tensor[*[n_samples]*i, n_restarts, 1, 1]

    # compute pairwise-distance d(a_i, a_j) for the diversity
    if K < 2:
        dist_reward = 0
    
    else:
        batch_as_dist = torch.cdist(actions.contiguous(), actions.contiguous(), p=1.0)
        # >>> n_samples x n_restarts x K x K

        batch_as_dist_triu = torch.triu(batch_as_dist)

        batch_as_dist_triu[batch_as_dist_triu > self.dist_threshold] = self.dist_threshold
        dist_reward = batch_as_dist_triu.sum((-1, -2)) / (K * (K - 1) / 2.0)  
        # >>> n_samples x n_restarts

    dist_reward = info["dist_weight"] * dist_reward

    # sum over samples from posterior predictive
    result = batch_yis - total_cost + dist_reward

    # compute the advantage
    # if c.baseline is not None: result = result - c.baseline
    result = result.squeeze()
    avg_result = result
    
    while len(avg_result.shape) > 1:
        avg_result = avg_result.mean(0)
        
    return avg_result


def compute_expectedloss_minmax(post_dist, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on batch_as.

    Args:
        batch_as: n_restarts x n_samples x n_actions x action_dim
    """
    assert batch_as.shape[2] == 2

    # Permute shape of batch_as to work with self.model.posterior correctly
    batch_as = torch.permute(batch_as, [1, 0, 2, 3])

    posterior = self.model.posterior(batch_as)
    # n_fs x n_samples x n_restarts x K x 1
    samples = self.sampler(posterior)
    val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
    val[:, :, :, 0] = -1 * val[:, :, :, 0]
    val = val.sum(dim=-1)  # n_fs x n_samples x n_restarts
    val = val.mean(dim=0)  # n_samples x n_restarts
    q_hes = val

    return q_hes.mean(dim=0)


def compute_expectedloss_twoval(post_dist, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on batch_as.

    Args:
        batch_as: n_restarts x n_samples x n_actions x action_dim
    """
    K = self.config.n_actions
    assert K == 2

    # Permute shape of batch_as to work with self.model.posterior correctly
    batch_as = torch.permute(batch_as, [1, 0, 2, 3])

    posterior = self.model.posterior(batch_as)
    # n_fs x n_samples x n_restarts x K x 1
    samples = self.sampler(posterior)
    val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
    val[:, :, :, 0] = -1 * torch.abs(val[:, :, :, 0] - self.val_tuple[0])
    val[:, :, :, 1] = -1 * torch.abs(val[:, :, :, 1] - self.val_tuple[1])
    val = val.sum(dim=-1)  # n_fs x n_samples x n_restarts
    val = val.mean(dim=0)  # n_samples x n_restarts
    q_hes = val

    close = True
    if close:
        batch_as_dist = torch.cdist(
            batch_as.contiguous(), batch_as.contiguous(), p=1.0
        )
        # n_samples x n_restarts x K x K
        batch_as_dist_triu = torch.triu(batch_as_dist)
        # n_samples x n_restarts
        dist_reward = -1 * \
            batch_as_dist_triu.sum((-1, -2)) / (K * (K - 1) / 2.0)
        dist_reward = 2 * dist_reward
        q_hes += dist_reward

    origin = True
    if origin:
        dist_origin_reward = -20 * \
            torch.linalg.norm(batch_as[:, :, 0, :], dim=-1)
        dist_origin_reward += -5 * \
            torch.linalg.norm(batch_as[:, :, 1, :], dim=-1)
        q_hes += dist_origin_reward

    return q_hes.mean(dim=0)


def compute_expectedloss_mvs(post_dist, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on batch_as.

    Args:
        batch_as: n_restarts x n_samples x n_actions x action_dim
    """

    K = self.config.n_actions

    # Permute shape of batch_as to work with self.model.posterior correctly
    batch_as = torch.permute(batch_as, [1, 0, 2, 3])

    posterior = self.model.posterior(batch_as)
    # n_fs x n_samples x n_restarts x K x 1
    samples = self.sampler(posterior)
    val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
    for idx in range(K):
        # val[:, :, :, idx] = -1 * torch.abs(val[:, :, :, idx] - self.val_tuple[idx])**2
        val[:, :, :, idx] = -1 * \
            torch.abs(val[:, :, :, idx] - self.val_tuple[idx])

    val = val.sum(dim=-1)  # n_fs x n_samples x n_restarts
    val = val.mean(dim=0)  # n_samples x n_restarts
    q_hes = val

    close = False
    if close:
        batch_as_dist = torch.cdist(
            batch_as.contiguous(), batch_as.contiguous(), p=1.0
        )
        # n_samples x n_restarts x K x K
        batch_as_dist_triu = torch.triu(batch_as_dist)
        # n_samples x n_restarts
        dist_reward = -1 * \
            batch_as_dist_triu.sum((-1, -2)) / (K * (K - 1) / 2.0)
        dist_reward = 1 * dist_reward
        q_hes += dist_reward

    origin = False
    if origin:
        dist_origin_reward = -20 * \
            torch.linalg.norm(batch_as[:, :, 0, :], dim=-1)
        dist_origin_reward += -5 * \
            torch.linalg.norm(batch_as[:, :, 1, :], dim=-1)
        q_hes += dist_origin_reward

    chain = True
    if chain:
        for idx in range(1, K):
            link_dist = batch_as[:, :, idx, :] - batch_as[:, :, idx - 1, :]
            # link_dist_reward = -0.01 * torch.linalg.norm(link_dist, dim=-1)
            link_dist_reward = -0.1 * torch.linalg.norm(link_dist, dim=-1)
            q_hes += link_dist_reward

    return q_hes.mean(dim=0)


def compute_expectedloss_levelset(post_dist, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on Actions.

    Args:
        support_points: n_actions x data_dim
        batch_as: n_restarts x n_samples x n_actions x action_dim
        where n_actions is the support size.
    """
    assert batch_as.shape[3] == 1

    # n_samples x n_restarts x n_actions
    batch_as = batch_as.squeeze(-1).permute([1, 0, 2])
    posterior = self.model.posterior(self.support_points)
    # n_fs x n_samples x n_restarts x n_actions x 1
    samples = self.sampler(posterior)
    val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x n_actions
    val = val.mean(dim=0)  # n_samples x n_restarts x n_actions
    q_hes = ((val - self.levelset_threshold) * batch_as).sum(
        dim=-1
    )  # n_samples x n_restarts
    return q_hes.mean(dim=0)


def compute_expectedloss_multilevelset(post_dist, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on Actions.

    Args:
        support_points: n_actions x data_dim
        batch_as: n_restarts x n_samples x n_actions x action_dim
        where n_actions is the support size.
    """
    assert batch_as.shape[3] == len(self.levelset_thresholds)
    # n_samples x n_restarts x n_actions x n_levelset
    batch_as = batch_as.permute([1, 0, 2, 3])
    posterior = self.model.posterior(self.support_points)
    # n_fs x n_samples x n_restarts x n_actions x 1
    samples = self.sampler(posterior)
    val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x n_actions
    val = val.mean(dim=0)  # n_samples x n_restarts x n_actions
    q_hes = 0
    for i, threshold in enumerate(self.levelset_thresholds):
        # n_samples x n_restarts
        q_hes += ((val - threshold) * batch_as[:, :, :, i]).sum(dim=-1)
    return q_hes.mean(dim=0)


def compute_expectedloss_expf(post_dist, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on batch_as.
    Args:
        batch_as: n_restarts x n_samples x n_actions x action_dim
    """

    K = self.config.n_actions

    # Permute shape of batch_as to work with self.model.posterior correctly
    batch_as = torch.permute(batch_as, [1, 0, 2, 3])

    # Draw samples from Normal distribution
    # --- Draw standard normal samples (of a certain shape)
    # std_normal = torch.normal(# TODO)
    # --- Transform, using batch_as, to get to correct means/stds/weights (encoded in batch_as)
    # TODO
    # --- Take function evals with self.model.posterior(samples)
    # TODO
    # --- Compute average of these function evals
    # TODO

    posterior = self.model.posterior(batch_as)
    # n_fs x n_samples x n_restarts x K x 1
    samples = self.sampler(posterior)
    val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
    val = val.sum(dim=-1)  # n_fs x n_samples x n_restarts
    val = val.mean(dim=0)  # n_samples x n_restarts

    batch_as_dist = torch.cdist(
        batch_as.contiguous(), batch_as.contiguous(), p=1.0)
    # n_samples x n_restarts x K x K
    batch_as_dist_triu = torch.triu(batch_as_dist)
    batch_as_dist_triu[
        batch_as_dist_triu > self.dist_threshold
    ] = self.dist_threshold
    dist_reward = batch_as_dist_triu.sum((-1, -2)) / (
        K * (K - 1) / 2.0
    )  # n_samples x n_restarts
    dist_reward = self.dist_weight * dist_reward

    q_hes = val + dist_reward
    q_hes = q_hes.squeeze()
    return q_hes.mean(dim=0)


def compute_expectedloss_pbest(post_dist, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on batch_as.

    Args:
        batch_as: n_restarts x n_samples x n_actions x action_dim
    """

    K = self.config.n_actions

    # Permute shape of batch_as to work with self.model.posterior correctly
    batch_as = torch.permute(batch_as, [1, 0, 2, 3])

    samples_rand_samp = self.sampler(self.posterior_rand_samp)
    maxes = torch.amax(samples_rand_samp, dim=(3, 4))

    posterior = self.model.posterior(batch_as)
    # out shape: n_fs x n_samples x n_restarts x K x 1
    samples = self.sampler(posterior)

    # out shape: n_fs x n_samples x n_restarts
    val = -1 * (maxes - samples.squeeze())
    val[val > -0.2] = -0.2
    val = val.mean(dim=0)  # out shape: n_samples x n_restarts

    q_hes = val
    q_hes = q_hes.squeeze()
    return q_hes.mean(dim=0)


def compute_expectedloss_bestofk(post_dist, actions, sampler, info) -> torch.Tensor:
    r"""Evaluate EHIGTopKInner on batch_as.

    Args:
        batch_as: n_restarts x n_samples x n_actions x action_dim
    """

    K = self.config.n_actions

    # Permute shape of batch_as to work with self.model.posterior correctly
    batch_as = torch.permute(batch_as, [1, 0, 2, 3])

    posterior = self.model.posterior(batch_as)
    # n_fs x n_samples x n_restarts x K x 1
    samples = self.sampler(posterior)
    val = samples.squeeze(-1)  # n_fs x n_samples x n_restarts x K
    val = val.amax(dim=-1)  # n_fs x n_samples x n_restarts
    val = val.mean(dim=0)  # n_samples x n_restarts

    q_hes = val
    q_hes = q_hes.squeeze()
    return q_hes.mean(dim=0)
