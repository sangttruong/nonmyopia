import pickle
import numpy as np
from _0_main import Parameters, make_env
from _5_evalplot import eval_and_plot
from draw_regrets import metric_files
import sys

datasets = sys.argv[1:]

algos = ["EI", "Non-myopic MST", "Non-myopic HES"]
algos_name = ["qEI", "qMSL", "HES"]
seeds = [1]

# Init environment
env = make_env(
    env_name=env_name,
    x_dim=local_parms.x_dim,
    bounds=local_parms.bounds
)
env = env.to(
    dtype=local_parms.torch_dtype,
    device=local_parms.device,
)
             
WM = SingleTaskGP(
            buffer["x"][:i],
            buffer["y"][:i],
            outcome_transform=Standardize(1),
            covar_module=parms.kernel,
        ).to(parms.device)

torch.save(buffer, f'{parms.save_dir}/buffer.pt')
            
torch.save(WM.state_dict(), f'{parms.save_dir}/world_model.pt')
           
real_loss, _ = eval_and_plot(
            func=env,
            wm=WM,
            cfg=parms,
            acqf=actor.acqf,
            buffer=buffer,
            next_x=next_x,
            actions=None,
            iteration=i,
        )