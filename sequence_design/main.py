import os
import pickle
import time
from argparse import ArgumentParser
from datetime import datetime
from typing import Any, Dict, Optional

import joblib
import torch
import wandb
import yaml
from actor import Actor
from bayesian_ridge import BayesianRidgeModel
from datasets import concatenate_datasets, Dataset, load_dataset
from peft import AutoPeftModelForCausalLM
from utils import (
    compute_ed,
    create_lookahead_sequences,
    ensure_dir,
    find_idx_in_dataset,
    get_embedding_from_server,
    import_protein_env,
    observe_value,
    random_sampling,
    read_yaml_to_dynamic_dataclass,
    set_seed,
    start_process,
)


def main(args: Optional[Dict[str, Any]] = None):
    wandb.init(project="nonmyopia-sequence")

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hes_ts_am.yaml")
    args = parser.parse_args()

    # Automatically create config class
    Config = read_yaml_to_dynamic_dataclass(args.config)
    config = Config()

    # Import protein environment
    _, INIT_SEQ, _, _, _ = import_protein_env(config.mutant_ver)

    # Set seed & create necessary directory
    set_seed(config.seed)

    # Initializing full dataset
    training_dataset = load_dataset(path=config.dataset, split="train")
    if "inputs_embeds" not in training_dataset.column_names:
        inputs_embeds_training = get_embedding_from_server(
            server_url=config.embedding_model, list_sequences=training_dataset["text"]
        )
        training_dataset = training_dataset.add_column(
            "inputs_embeds", inputs_embeds_training
        )

    # testing_dataset = load_dataset(path=config.dataset, split="test")
    # if "inputs_embeds" not in testing_dataset.column_names:
    #     inputs_embeds_testing = get_embedding_from_server(
    #         server_url=config.embedding_model, list_sequences=testing_dataset["text"]
    #     )
    #     testing_dataset = testing_dataset.add_column(
    #         "inputs_embeds", inputs_embeds_testing
    #     )

    # Initializing oracle model
    print("Creating Oracle...")
    if os.path.exists(f"{config.oracle_path}/model.joblib"):
        oracle = joblib.load(f"{config.oracle_path}/model.joblib")
    else:
        X_train = torch.tensor(training_dataset["inputs_embeds"])
        y_train = torch.tensor(training_dataset["reward"]).reshape(-1, 1)
        oracle = BayesianRidgeModel(X_train, y_train)
        ensure_dir(f"{config.oracle_path}")
        joblib.dump(oracle, f"{config.oracle_path}/model.joblib")

    # Initializing training buffer
    # Randomly sample from training dataset
    initial_dataset = random_sampling(
        training_dataset, num_samples=config.initinal_sequences, is_init=True
    )
    # Query Oracle for y
    initial_dataset_reward = observe_value(
        oracle,
        torch.Tensor(initial_dataset["inputs_embeds"]),
        compute_ed(INIT_SEQ, initial_dataset["text"]),
        config.fn_ver,
    )
    initial_dataset = (
        initial_dataset.remove_columns("reward")
        .add_column("reward", initial_dataset_reward)
        .cast(initial_dataset.features)
    )

    # Query Oracle for y
    # testing_dataset_reward = observe_value(
    #     oracle,
    #     torch.Tensor(testing_dataset["inputs_embeds"]),
    #     compute_ed(INIT_SEQ, testing_dataset["text"]),
    #     config.fn_ver,
    # )
    # testing_dataset = (
    #     testing_dataset.remove_columns("reward")
    #     .add_column("reward", testing_dataset_reward)
    #     .cast(testing_dataset.features)
    # )
    # Random choose sequences with reward < 2.0 as inital sequence
    # initial_sequences = random_sampling(
    #     testing_dataset, num_samples=config.n_sequences, constrained_reward=1.0
    # )
    init_idx = find_idx_in_dataset(training_dataset, "text", INIT_SEQ)
    initial_sequences = training_dataset.select([init_idx])
    initial_rewards = observe_value(
        oracle,
        torch.Tensor(initial_sequences["inputs_embeds"]),
        compute_ed(INIT_SEQ, initial_sequences["text"]),
        config.fn_ver,
    )
    initial_sequences = (
        initial_sequences.remove_columns("reward")
        .add_column("reward", initial_rewards)
        .cast(initial_sequences.features)
    )

    # Merge initial_sequences to initial_dataset
    initial_dataset = concatenate_datasets([initial_dataset, initial_sequences])

    if config.continue_iter:
        with open(f"{config.output_dir}/buffer.pkl", "rb") as f:
            buffer = pickle.load(f)
        buffer_length = len(buffer["x"])
        assert (
            buffer_length == config.continue_iter + 1
        ), f"Maximal continue iteration = {buffer_length - 1}"
    else:
        config.continue_iter = 0
        buffer = {
            "dataset": initial_dataset,
            "x": [initial_sequences["text"]],
            "y": [initial_sequences["reward"]],
        }

        # Ensure ckpt folder
        ensure_dir(f"{config.output_dir}")

        # Save buffer
        with open(f"{config.output_dir}/buffer.pkl", "wb") as f:
            pickle.dump(buffer, f)

    actor = Actor(config)

    # -------------------------------------------------------

    # Create SFT dataset for pretraining Policy
    timestamp = datetime.today().isoformat()
    sft_ds_name = f"sft_{config.mutant_ver}_{config.fn_ver}_n{config.n_sequences}_lah{config.algo_lookahead_steps}_s{config.seed}"
    if not os.path.exists(f"data/{sft_ds_name}"):
        sft_ds = create_lookahead_sequences(
            config, actor.policy.tokenizer, oracle, initial_sequences, config.fn_ver
        )
        sft_ds.to_csv(f"data/{sft_ds_name}/{sft_ds_name}_train.csv")
        sft_ds.to_csv(f"data/{sft_ds_name}/{sft_ds_name}_test.csv")

    # SFT Training for Policy
    output_dir = os.path.join(
        f"ckpts/sft_model_{config.mutant_ver}_{config.fn_ver}_n{config.n_sequences}_lah{config.algo_lookahead_steps}_s{config.seed}"
    )
    if not os.path.exists(output_dir):
        with open("configs/sft.yaml", "r", encoding="utf8") as stream:
            loaded_configs = yaml.safe_load(stream)
            loaded_configs["output_dir"] = output_dir
            loaded_configs["model_name_or_path"] = config.policy_model_name_or_path
            loaded_configs["dataset_name"] = f"data/{sft_ds_name}"

        training_config_file = f"configs/sft_{timestamp}.yaml"
        with open(training_config_file, "w", encoding="utf8") as stream:
            yaml.dump(
                loaded_configs, stream, default_flow_style=False, allow_unicode=True
            )
        start_process(
            f"CUDA_VISIBLE_DEVICES={config.ppo_gpu} accelerate launch --main_process_port {config.main_process_port} "
            "--config_file configs/single_config.yaml "
            "sft.py "
            f"--config {training_config_file}"
        )

        # Remove temp config file
        os.system(f"rm -rf {training_config_file}")

        # Merge Lora adapter
        if loaded_configs["use_peft"]:
            model = AutoPeftModelForCausalLM.from_pretrained(output_dir)
            model = model.merge_and_unload().to(torch.bfloat16)
            model.save_pretrained(output_dir)
            del model
            model = None

    # -------------------------------------------------------

    # Startign BO loop
    for i in range(config.continue_iter, config.algo_n_iterations):
        print(f"Starting BO loop #{i}")

        # Ensure ckpt folder
        ensure_dir(f"{config.output_dir}/{i}")

        X_train = buffer["dataset"].data["inputs_embeds"].to_pylist()
        y_train = buffer["dataset"].data["reward"].to_pylist()
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train).reshape(-1, 1)

        # Revert to original range for training RM
        y_train = -torch.log(6 / (y_train + 3) - 1)
        reward_model = BayesianRidgeModel(X_train, y_train)
        joblib.dump(
            reward_model,
            os.path.join(f"{config.output_dir}/{i}/reward_model.joblib"),
        )

        # -------------------------------------------------------

        # Adjusting the lookahead steps
        if actor.algo_lookahead_steps > 0 and (
            config.algo_n_iterations - i - 1 < actor.algo_lookahead_steps
        ):
            actor.algo_lookahead_steps = config.algo_n_iterations - i - 1

        # Create prompt dataset for policy training
        prompt_dataset = actor.create_dataset(prevX=buffer["x"], prevY=buffer["y"])

        # Train new policy with rolled out dataset
        query_start_time = time.time()
        actor.train_policy(iteration=i, dataset=prompt_dataset, mutant_ver=config.mutant_ver, fn_ver=config.fn_ver)

        # -------------------------------------------------------

        # Get the next X
        actor.policy.load_inference(iteration=i)
        iter_start_time = time.time()

        next_X = actor.query(
            iteration=i,
            prevX=buffer["x"],
            prevY=buffer["y"],
            reward_model=reward_model,
            n_restarts=config.n_restarts,
        )

        iter_end_time = time.time()

        print(f"Iteration {i} took {iter_end_time - iter_start_time} seconds")
        actor.policy.unload_inference()

        # Query Oracle for y
        observed_emb = get_embedding_from_server(
            server_url=config.embedding_model, list_sequences=next_X
        )
        next_y = observe_value(
            oracle,
            torch.tensor(observed_emb),
            compute_ed(INIT_SEQ, next_X),
            config.fn_ver,
        )
        query_end_time = time.time()

        buffer["x"].append(next_X)
        buffer["y"].append(next_y)
        print("Next X", next_X)
        print("Next y", next_y)

        # -------------------------------------------------------

        # Logging
        log_dict = {f"y{k}": v for k, v in enumerate(next_y)}
        log_dict.update({"runtime": query_end_time - query_start_time})
        wandb.log(log_dict)

        # Merge dataset for reward_model
        observed = Dataset.from_dict(
            {"text": next_X, "inputs_embeds": observed_emb, "reward": next_y}
        )
        buffer["dataset"] = concatenate_datasets([buffer["dataset"], observed])

        # Save buffer
        with open(f"{config.output_dir}/buffer.pkl", "wb") as f:
            pickle.dump(buffer, f)

        # Remove unneccessary checkpoints
        if i > 0:
            ckpt_files = os.path.join(
                config.output_dir, f"{i-1}", "model-*.safetensors"
            )
            os.system(f"rm -rf {ckpt_files}")

    # WandB end
    wandb.finish()


if __name__ == "__main__":
    main()
