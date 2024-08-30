import os
import pickle
import time
from argparse import ArgumentParser
from typing import Any, Dict, Optional

import joblib
import torch
import wandb

from actor import Actor
from bayesian_ridge import BayesianRidgeModel
from datasets import concatenate_datasets, Dataset, load_dataset
from embed_text_package.embed_text import Embedder
from torch.utils.data import DataLoader
from utils import ensure_dir, random_sampling, read_yaml_to_dynamic_dataclass, set_seed


def main(args: Optional[Dict[str, Any]] = None):
    wandb.init(project="nonmyopia-sequence")

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hes_ts_am.yaml")
    args = parser.parse_args()

    # Automatically create config class
    Config = read_yaml_to_dynamic_dataclass(args.config)
    config = Config()

    # Set seed & create necessary directory
    set_seed(config.seed)

    # Initializing embedding model
    embedder = Embedder()
    embedder.load(config.embedding_model_name_or_path)

    # Initializing full dataset
    training_dataset = load_dataset(path=config.dataset, split="train")
    if "inputs_embeds" not in training_dataset.column_names:
        inputs_embeds_training = (
            embedder.get_embeddings(
                DataLoader(training_dataset, batch_size=1),
                config.embedding_model_name_or_path,
                ["text"],
            )
            .data["text"]
            .to_pylist()
        )
        training_dataset = training_dataset.add_column(
            "inputs_embeds", inputs_embeds_training
        )

    testing_dataset = load_dataset(path=config.dataset, split="test")
    if "inputs_embeds" not in testing_dataset.column_names:
        inputs_embeds_testing = (
            embedder.get_embeddings(
                DataLoader(testing_dataset, batch_size=1),
                config.embedding_model_name_or_path,
                ["text"],
            )
            .data["text"]
            .to_pylist()
        )
        testing_dataset = testing_dataset.add_column(
            "inputs_embeds", inputs_embeds_testing
        )

    # Initializing oracle model
    X_train = torch.tensor(training_dataset["inputs_embeds"])
    y_train = torch.tensor(training_dataset["reward"]).reshape(-1, 1)
    oracle = BayesianRidgeModel(X_train, y_train)

    # Initializing training buffer
    # Randomly sample from training dataset
    initial_dataset = random_sampling(
        training_dataset, num_samples=config.initinal_sequences
    )
    # Query Oracle for y
    initial_dataset_reward = oracle.predict(
        torch.Tensor(initial_dataset["inputs_embeds"])
    ).tolist()
    initial_dataset = (
        initial_dataset.remove_columns("reward")
        .add_column("reward", initial_dataset_reward)
        .cast(initial_dataset.features)
    )

    # Random choose sequences with reward < 2.0 as inital sequence
    initial_sequences = random_sampling(
        testing_dataset, num_samples=config.n_sequences, constrained_reward=1.5
    )
    # Query Oracle for y
    initial_sequences_reward = oracle.predict(
        torch.Tensor(initial_sequences["inputs_embeds"])
    ).tolist()
    initial_sequences = (
        initial_sequences.remove_columns("reward")
        .add_column("reward", initial_sequences_reward)
        .cast(initial_sequences.features)
    )

    # Merge initial_sequences to initial_dataset
    initial_dataset = concatenate_datasets([initial_dataset, initial_sequences])

    buffer = {
        "dataset": initial_dataset,
        "x": [initial_sequences["text"]],
        "y": [initial_sequences["reward"]],
    }

    actor = Actor(config)

    # Startign BO loop
    for i in range(config.algo_n_iterations):
        print(f"Starting BO loop #{i}")

        # Ensure ckpt folder
        ensure_dir(f"{config.output_dir}/{i}")

        X_train = buffer["dataset"].data["inputs_embeds"].to_pylist()
        y_train = buffer["dataset"].data["reward"].to_pylist()
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train).reshape(-1, 1)

        reward_model = BayesianRidgeModel(X_train, y_train)
        joblib.dump(
            reward_model,
            os.path.join(f"{config.output_dir}/{i}/reward_model.joblib"),
        )
        # -------------------------------------------------------

        # Adjusting the lookahead steps
        if actor.algo_lookahead_steps > 1 and (
            config.algo_n_iterations - i < actor.algo_lookahead_steps
        ):
            actor.algo_lookahead_steps -= 1

        # Create prompt dataset for policy training
        prompt_dataset = actor.create_dataset(prevX=buffer["x"], prevY=buffer["y"])

        # Train new policy with rolled out dataset
        query_start_time = time.time()
        # We must unload embedder here because we will load separated reward server
        embedder.unload()
        actor.train_policy(iteration=i, dataset=prompt_dataset)
        embedder.load(config.embedding_model_name_or_path)

        # Get the next X
        actor.policy.load_inference(iteration=i)
        iter_start_time = time.time()

        next_X = actor.query(
            prevX=buffer["x"],
            prevY=buffer["y"],
            embedder=embedder,
            reward_model=reward_model,
            n_restarts=config.n_restarts,
        )

        iter_end_time = time.time()

        print(f"Iteration {i} took {iter_end_time - iter_start_time} seconds")
        actor.policy.unload_inference()

        # Query Oracle for y
        observed = Dataset.from_dict({"text": next_X})
        observed_emb = (
            embedder.get_embeddings(
                DataLoader(observed, batch_size=1),
                config.embedding_model_name_or_path,
                ["text"],
            )
            .data["text"]
            .to_pylist()
        )
        next_y = oracle.predict(torch.tensor(observed_emb)).tolist()
        query_end_time = time.time()

        buffer["x"].append(next_X)
        buffer["y"].append(next_y)
        print("Next X", next_X)
        print("Next y", next_y)

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

    # WandB end
    wandb.finish()


if __name__ == "__main__":
    main()
