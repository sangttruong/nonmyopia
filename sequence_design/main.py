import time
import pickle
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional
from datasets import concatenate_datasets, Dataset, Features, Value

from llmtuner.extras.callbacks import LogCallback
from llmtuner.data.loader import load_single_dataset
from llmtuner.data.parser import get_dataset_list
from llmtuner.data.utils import merge_dataset

from actor import Actor
from hparams import get_bo_args
from bayesian_ridge import BayesianRidgeModel
from embed_text_package.embed_text import Embedder
from utils import (
    set_seed,
    random_sampling,
    load_bayesian_ridge_model
)


def main(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    surr_model_args, oracle_model_args, policy_model_args, data_args, training_args, \
        finetuning_args, generating_args, bo_args = get_bo_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks
    # Set seed
    set_seed(training_args.seed)

    # Initializing full dataset
    with training_args.main_process_first(desc="load training dataset"):
        all_datasets = []
        data_args.split = "train"
        for dataset_attr in get_dataset_list(data_args):
            all_datasets.append(load_single_dataset(
                dataset_attr, policy_model_args, data_args))
        training_dataset = merge_dataset(
            all_datasets, data_args, training_args)
        training_dataset = training_dataset.cast(
            Features({'text': Value(dtype='string'),
                     'reward': Value(dtype='float32')})
        )

    with training_args.main_process_first(desc="load testing dataset"):
        all_datasets = []
        data_args.split = "validation"
        for dataset_attr in get_dataset_list(data_args):
            all_datasets.append(load_single_dataset(
                dataset_attr, policy_model_args, data_args))
        testing_dataset = merge_dataset(
            all_datasets, data_args, training_args)
        testing_dataset = testing_dataset.cast(
            Features({'text': Value(dtype='string'),
                     'reward': Value(dtype='float32')})
        )

    # Initializing embedding model
    embeder = Embedder()
    embeder.load(oracle_model_args.model_name_or_path)

    # Initializing oracle model
    X_train = embeder.get_embeddings(
        DataLoader(training_dataset, batch_size=1), 
        oracle_model_args.model_name_or_path, 
        ["text"]
    ).data["text"]
    y_train = torch.tensor(train_dataset["rewards"]).reshape(-1, 1)
    oracle = BayesianRidgeModel(X_train, y_train)

    # Initializing training buffer
    # Randomly sample from training dataset
    initial_dataset = random_sampling(
        training_dataset, num_samples=bo_args.initinal_sequences)
    # Query Oracle for y
    initial_emb = embeder.get_embeddings(
        DataLoader(initial_dataset, batch_size=1), 
        oracle_model_args.model_name_or_path, 
        ["text"]
    ).data["text"]
    initial_dataset_reward = oracle.predict(initial_emb.float().cpu().detach().numpy())
    initial_dataset = initial_dataset.remove_columns("reward").add_column(
        "reward", initial_dataset_reward).cast(initial_dataset.features)

    # Random choose sequences with reward < 2.0 as inital sequence
    initial_sequences = random_sampling(
        testing_dataset, num_samples=bo_args.n_sequences, constrained_reward=2.0)
    # Query Oracle for y
    initial_seq_emb = embeder.get_embeddings(
        DataLoader(initial_sequences, batch_size=1), 
        oracle_model_args.model_name_or_path, 
        ["text"]
    ).data["text"]
    initial_sequences_reward = oracle.predict(initial_seq_emb.float().cpu().detach().numpy())
    initial_sequences = initial_sequences.remove_columns("reward").add_column(
        "reward", initial_sequences_reward).cast(initial_sequences.features)

    # Merge initial_sequences to initial_dataset
    initial_dataset = concatenate_datasets(
        [initial_dataset, initial_sequences])

    buffer = {
        "dataset": initial_dataset,
        "x": [initial_sequences["text"]],
        "y": [initial_sequences["reward"]]
    }

    actor = Actor(bo_args, policy_model_args, finetuning_args,
                  data_args, training_args, generating_args)

    # Startign BO loop
    for i in range(bo_args.algo_n_iterations):
        # Warming up reward models
        dataset_emb = embeder.get_embeddings(
            DataLoader(buffer["dataset"], batch_size=1), 
            oracle_model_args.model_name_or_path, 
            ["text"]
        )

        X_train = dataset_emb.data["text"].to_pylist()
        y_train = buffer["dataset"].data["rewards"].to_pylist()
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train).reshape(-1, 1)

        surr_model = BayesianRidgeModel(X_train, y_train)
        # -------------------------------------------------------


        # Adjusting the lookahead steps
        if actor.algo_lookahead_steps > 1 and (
            bo_args.algo_n_iterations - i < actor.algo_lookahead_steps
        ):
            actor.algo_lookahead_steps -= 1

        # Rollout training dataset for policy
        rollout_dataset = actor.rollout(
            surr_model,
            buffer["x"][-1],
            n_sequences=bo_args.rollout_sequences
        )

        # Train new policy with rolled out dataset
        actor.load_policy(iteration=i)
        actor.train_policy(dataset=rollout_dataset, data_args=data_args)
        actor.unload_policy()

        # Get the next X
        server_process = actor.load_policy_inference()
        iter_start_time = time.time()

        next_X = actor.query(
            prevX=buffer["x"],
            prevY=buffer["y"],
            reward_model=surr_model,
            n_restarts=bo_args.n_restarts
        )

        iter_end_time = time.time()
        actor.unload_policy_inference(server_process)
        print(f"Iteration {i} took {iter_end_time - iter_start_time} seconds")

        # Query Oracle for y
        observed = Dataset.from_dict({"text": next_X})
        observed_emb = embeder.get_embeddings(
            DataLoader(observed, batch_size=1), 
            oracle_model_args.model_name_or_path, 
            ["text"]
        ).data["text"]
        next_y = oracle.predict(observed_emb.float().cpu().detach().numpy())

        buffer["x"].append(next_X)
        buffer["y"].append(next_y)
        print("Next X", next_X)
        print("Next y", next_y)

        # Merge dataset for surr_model
        observed = Dataset.from_dict({"text": next_X, "reward": next_y})
        observed = observed.cast(
            Features({'text': Value(dtype='string'),
                     'reward': Value(dtype='float32')})
        )
        buffer["dataset"] = concatenate_datasets([buffer["dataset"], observed])

        # Save buffer
        with open("results/buffer.pkl", "wb") as f:
            pickle.dump(buffer, f)


if __name__ == '__main__':
    main()
