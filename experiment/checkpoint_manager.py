from pathlib import Path
from argparse import Namespace
import dill as pickle

def pickle_trial_info(config, data, eval_metric_list, optimal_action_list):
    """Save trial info as a pickle in directory specified by config."""
    # Build trial info Namespace
    data = Namespace(x=data.x.cpu().detach().numpy(),
                     y=data.y.cpu().detach().numpy())
    trial_info = Namespace(
        config=config, 
        data=data, 
        eval_metric_list=eval_metric_list, 
        optimal_action_list=optimal_action_list
    )

    # Pickle trial info
    dir_path = Path(str(config.save_dir))
    file_path = dir_path / "trial_info.pkl"
    with open(str(file_path), "wb") as file_handle:
        pickle.dump(trial_info, file_handle)


def make_save_dir(config):
    """Create save directory safely (without overwriting directories), using config."""
    init_dir_path = Path(config.save_dir)
    dir_path = Path(str(init_dir_path))

    for i in range(50):
        try:
            dir_path.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            dir_path = Path(str(init_dir_path) + "_" + str(i).zfill(2))

    config.save_dir = str(dir_path)
    print(f"Created save_dir: {config.save_dir}")
    
    # Save config to save_dir as parameters.json
    config_path = dir_path / "parameters.json"
    with open(str(config_path), "w") as file_handle:
        config_dict = str(config)
        file_handle.write(config_dict)
    
