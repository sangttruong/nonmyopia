from models.mykg import MyqKnowledgeGradient, initialize_action_tensor_kg
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.acquisition import (
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.sampling.normal import SobolQMCNormalSampler
import torch


class Actor:
    def __init__(self, parms, seed, algo):
        self.parms = parms
        self.seed = seed
        self.algo = algo 

        # Initialize model
        mll_hes, model_hes = initialize_model(
            data, covar_module=ScaleKernel(base_kernel=RBFKernel())
        )

        # Fit the model
        if not parms.learn_hypers:
            print(
                f"config.learn_hypers={parms.learn_hypers}, using hypers from config.hypers"
            )
            model_hes.covar_module.base_kernel.lengthscale = [
                [parms.hypers["ls"]]]
            # NOTE: GPyTorch outputscale should be set to the SynthFunc alpha squared
            model_hes.covar_module.outputscale = parms.hypers["alpha"] ** 2
            model_hes.likelihood.noise_covar.noise = [parms.hypers["sigma"]]

            model_hes.covar_module.base_kernel.raw_lengthscale.requires_grad_(
                False)
            model_hes.covar_module.raw_outputscale.requires_grad_(False)
            model_hes.likelihood.noise_covar.raw_noise.requires_grad_(False)

        fit_gpytorch_model(mll_hes)
        print_model_hypers(model_hes)


    def print_model_hypers(model):
        """Print current hyperparameters of GP model."""
        raw_hypers_str = (
            "\n*Raw GP hypers: "
            f"\nmodel.covar_module.base_kernel.raw_lengthscale={model.covar_module.base_kernel.raw_lengthscale.tolist()}"
            f"\nmodel.covar_module.raw_outputscale={model.covar_module.raw_outputscale.tolist()}"
            f"\nmodel.likelihood.noise_covar.raw_noise={model.likelihood.noise_covar.raw_noise.tolist()}"
        )
        actual_hypers_str = (
            "\n*Actual GP hypers: "
            f"\nmodel.covar_module.base_kernel.lengthscale={model.covar_module.base_kernel.lengthscale.tolist()}"
            f"\nmodel.covar_module.outputscale={model.covar_module.outputscale.tolist()}"
            f"\nmodel.likelihood.noise_covar.noise={model.likelihood.noise_covar.noise.tolist()}"
        )
        print(raw_hypers_str)
        print(actual_hypers_str + "\n")


    def initialize_model(data, state_dict=None, covar_module=None):
        model = SingleTaskGP(data.x, data.y, covar_module=covar_module).to(data.x)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model


    def query(self):
        if parms.algo in ["hes_mc", "hes_vi"]:
            min_loss_, next_x = optimize_hes(hes, iteration)
        elif parms.algo == "rs":
            next_x = optimize_rs(parms)
        elif parms.algo == "us":
            next_x = optimize_us(model_hes, parms)
        elif parms.algo == "kg" or parms.algo == "kgtopk":
            pass 

        elif parms.algo == "random":
            next_x = parms.bounds[0] + (
                parms.bounds[1] - parms.bounds[0]
            ) * torch.rand([parms.n_candidates, parms.n_dim], device=parms.device)
            

        elif parms.algo in ["qEI", "qPI", "qSR", "qUCB"]:
            sampler = SobolQMCNormalSampler(
                sample_shape=parms.n_samples, seed=0, resample=False
            )
            if parms.algo == "qEI":
                acq_function = qExpectedImprovement(
                    model_hes, best_f=data.y.max(), sampler=sampler
                )
            elif parms.algo == "qPI":
                acq_function = qProbabilityOfImprovement(
                    model_hes, best_f=data.y.max(), sampler=sampler
                )
            elif parms.algo == "qSR":
                acq_function = qSimpleRegret(model_hes, sampler=sampler)
            elif parms.algo == "qUCB":
                acq_function = qUpperConfidenceBound(
                    model_hes, beta=0.1, sampler=sampler
                )

            # to keep the restart conditions the same
            torch.manual_seed(seed=0)
            bounds = torch.tensor(
                [
                    [parms.bounds[0]] * parms.n_dim,
                    [parms.bounds[1]] * parms.n_dim,
                ]
            ).to(parms.device).double()
            next_x, _ = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=parms.n_restarts,
                raw_samples=1000,
                options={},
            )


        return next_x
    
    def optimize_rs(config):
        """Optimize random search (rs) acquisition function, return next_x."""
        data_x = uniform_random_sample_domain(config.domain, 1)
        next_x = data_x[0].reshape(1, -1)
        return next_x


    def optimize_us(model, config):
        """Optimize uncertainty sampling (us) acquisition function, return next_x."""
        n_acq_opt_samp = 500
        data_x = uniform_random_sample_domain(config.domain, n_acq_opt_samp)
        acq_values = model(data_x).variance
        best = torch.argmax(acq_values.view(-1), dim=0)
        next_x = data_x[best].reshape(1, -1)
        return next_x


    def optimize_kg(batch_x0s, batch_a1s, model, sampler, config, iteration):
        """Optimize knowledge gradient (kg) acquisition function, return next_x."""
        if not config.n_dim == config.n_dim_action:
            batch_a1s = initialize_action_tensor_kg(config)

        qkg = MyqKnowledgeGradient(model, config, sampler)

        optimizer = torch.optim.Adam([batch_x0s, batch_a1s], lr=config.acq_opt_lr)
        for i in range(config.acq_opt_iter):
            losses = -qkg(batch_x0s, batch_a1s)
            loss = losses.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_x0s.data.clamp_(config.bounds[0], config.bounds[1])
            batch_a1s.data.clamp_(config.bounds_action[0], config.bounds_action[1])
            if (i + 1) % (config.acq_opt_iter // 5) == 0 or i == config.acq_opt_iter - 1:
                print(iteration, i + 1, loss.item())
        acq_values = qkg(batch_x0s, batch_a1s)
        best = torch.argmax(acq_values.view(-1), dim=0)
        next_x = batch_x0s[best]
        return next_x


    def optimize_hes(hes, iteration):
        c = hes.config
        params = []
        if c.algo == "hes_vi":
            params_ = [
                hes.maps_i[i][j].parameters()
                for j in range(c.n_restarts)
                for i in range(c.lookahead_steps + 1)
            ]

            for i in range(len(params_)):
                params += list(params_[i])
        elif c.algo == "hes_mc":
            for i in range(c.lookahead_steps + 1):
                params.append(hes.po[i])

        # Test number of params in VI or MC
        if iteration == c.start_iter:
            total_num_params = sum(p.numel() for p in params if p.requires_grad)
            print(f"Total params: {total_num_params}")
            if c.algo == "hes_vi":
                nn_params = 0
                for i in range(0, c.lookahead_steps + 1):
                    nn_params += sum(p.numel()
                                    for p in hes.maps_i[i][0].parameters())
                assert c.n_restarts * nn_params == total_num_params
            elif c.algo == "hes_mc":
                params_A = 0
                params_B = 0
                for i in range(c.lookahead_steps + 1):
                    if i == 0 or i == 1:
                        params_A += c.n_samples ** i
                    else:
                        tmp = c.n_samples
                        for j in range(1, i):
                            tmp *= math.ceil(c.n_samples / (c.decay_factor ** j))
                        if i == c.lookahead_steps:
                            params_B = c.n_dim_action * c.n_actions * tmp
                        else:
                            params_A += tmp
                params_A = c.n_dim * params_A
                assert c.n_restarts * (params_A + params_B) == total_num_params

        optim = torch.optim.Adam(params, lr=c.acq_opt_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=c.T_max, eta_min=c.eta_min
        )

        losses = []
        lrs = []
        patient = c.max_patient
        min_loss = float("inf")
        print("start optimizing acquisition function")
        for _ in tqdm(range(c.acq_opt_iter)):
            loss = -hes().sum()
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            if c.algo == "hes_mc":
                for i in range(c.lookahead_steps + 1):
                    hes.po[i] = torch.tanh(params[i])

            lrs.append(scheduler.get_last_lr())
            losses.append(loss.cpu().detach().numpy())

            if loss < min_loss:
                min_loss = loss
                patient = c.max_patient
                acq_values = hes()  # get acq values for all restarts
                # best result from all restarts
                best_restart = torch.argmax(acq_values)
                next_x = hes.po[0][best_restart]

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
        plt.savefig(f"{c.save_dir}/{c.algo}/acq_opt{iteration}.png",
                    bbox_inches="tight")

        return min_loss, next_x
        
