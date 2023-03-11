#!/usr/bin/env python

import torch

__author__ = ""
__copyright__ = "Copyright 2022, Stanford University"


def baseline(config, mc_opt_params, model, sampler, batch_yis):
    if config.baseline:
        # 1. compute the target hes for learning
        K = config.n_actions

        # permute shape of batch_as to work with model.posterior correctly
        if config.lookahead_steps == 1 or len(mc_opt_params["batch_as"].shape) == 4:
            batch_as = torch.permute(mc_opt_params["batch_as"], [1, 0, 2, 3])
        elif config.lookahead_steps == 2:
            batch_as = torch.permute(
                mc_opt_params["batch_as"], [1, 2, 0, 3, 4])
        elif config.lookahead_steps == 3:
            batch_as = torch.permute(
                mc_opt_params["batch_as"], [1, 2, 3, 0, 4, 5])
        else:
            raise NotImplemented

        posterior = model.posterior(batch_as)

        # TensorType["n_fs", "n_samples", "n_restarts", K, 1]
        f_as = sampler(posterior)
        # TensorType["n_fs", "n_samples", "n_restarts"]
        sum_f_as = f_as.squeeze(-1).sum(dim=-1)
        # TensorType["n_samples", "n_restarts"]
        avg_f_as = sum_f_as.mean(dim=0)

        if K >= 2:
            batch_as_dist = torch.cdist(
                mc_opt_params["batch_as"].contiguous(),
                mc_opt_params["batch_as"].contiguous(),
                p=1.0,
            )
            # TensorType["n_samples", "n_restarts", K, K]
            batch_as_dist_triu = torch.triu(batch_as_dist)
            batch_as_dist_triu[
                batch_as_dist_triu > config.dist_threshold
            ] = config.dist_threshold
            # TensorType["n_samples", "n_restarts"]
            dist_reward = batch_as_dist_triu.sum(
                (-1, -2)) / (K * (K - 1) / 2.0)
        else:
            dist_reward = 0.0
        dist_reward = config.dist_weight * dist_reward
        batch_hes_target = avg_f_as + dist_reward

        D1_xdim = [config.n_restarts,
                   config.n_samples, -1, config.n_dim_design]
        init_x = config.init_data.x[None, None, :].expand(*D1_xdim)
        batch_xs = mc_opt_params["batch_xs"][-1, None, :].expand(*D1_xdim)
        D1_x = torch.cat((init_x, batch_xs), dim=-2)
        D1_ydim = [config.n_restarts, config.n_samples, -1, 1]
        init_y = config.init_data.y[None, None, :].expand(*D1_ydim)
        batch_ys = batch_yis["batch_y0s"][-1, -1, None, :].expand(*D1_ydim)
        D1_y = torch.cat((init_y, batch_ys), dim=-2)
        D1 = torch.cat((D1_x, D1_y), dim=-1)
        D1 = D1.reshape(config.n_restarts, config.n_samples, -1).float()
        input_h = D1

        assert config.lookahead_steps <= 3
        raise NotImplemented

        # 3. learning to map D1 --> H[ p(f|D1) ]
        learnable_params = []
        for i in range(config.n_restarts):
            learnable_params += list(batch_mlp_h[i].parameters())
        optimizer = torch.optim.Adam(learnable_params, lr=0.1)

        optimization_objective = float("inf")
        epoch = 0
        print("Baseline fitting with MSE loss")
        while optimization_objective > 0.01 and epoch < 200:
            optimizer.zero_grad()
            batch_hes_predict = [
                batch_mlp_h[i](input_h[i]) for i in range(config.n_restarts)
            ]
            batch_hes_predict = torch.stack(batch_hes_predict).squeeze()
            n_dim_batch_hes_predict = len(batch_hes_predict.shape)
            batch_hes_predict = batch_hes_predict.permute(
                *[i for i in range(1, n_dim_batch_hes_predict)], 0
            )
            optimization_objective = (
                ((batch_hes_predict - batch_hes_target) ** 2).sum(-1).mean()
            )
            optimization_objective.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print("{} {:.2f}".format(epoch, optimization_objective.item()))
            epoch = epoch + 1

        # 4. forward
        batch_baselines = [batch_mlp_h[i](input_h[i])
                           for i in range(config.n_restarts)]
        batch_baselines = torch.stack(batch_baselines).squeeze()
        n_dim_batch_baselines = len(batch_hes_predict.shape)
        batch_baselines = batch_baselines.permute(
            *[i for i in range(1, n_dim_batch_baselines)], 0
        )
        return batch_baselines

    else:
        return None
