import os
import time
import torch
import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from datasets import concatenate_datasets, Dataset, Features, Value

from src.llmtuner.hparams import get_bo_args
from src.llmtuner.extras.callbacks import LogCallback
from src.llmtuner.train.tuner import export_model

from src.llmtuner.data.loader import load_single_dataset
from src.llmtuner.model import load_model, load_tokenizer
from src.llmtuner.data.parser import get_dataset_list
from src.llmtuner.data.utils import checksum, merge_dataset

from oracle import Oracle
from world_model import WorldModel
from actor import Actor
from utils import (
    set_seed,
    get_dataset_embedding,
    random_sampling,
    fix_oracle_model_args,
    fix_policy_model_args,
    fix_wm_model_args,
    fix_finetuning_policy_args,
    fix_finetuning_wm_args,
)


def main(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    wm_model_args, oracle_model_args, policy_model_args, data_args, training_args, \
        wm_finetuning_args, policy_finetuning_args, generating_args, bo_args = get_bo_args(
            args)
    callbacks = [LogCallback()] if callbacks is None else callbacks
    # Set seed
    set_seed(training_args.seed)

    # Fixing args
    fix_wm_model_args(wm_model_args)
    fix_oracle_model_args(oracle_model_args)
    fix_policy_model_args(policy_model_args)
    fix_finetuning_wm_args(wm_finetuning_args)
    fix_finetuning_policy_args(policy_finetuning_args)

    # Initializing models
    oracle = Oracle(oracle_model_args, wm_finetuning_args)
    world_model = WorldModel(wm_model_args, wm_finetuning_args)

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

    # Initializing training buffer
    oracle.load()

    # Randomly sample from training dataset
    initial_dataset = random_sampling(
        training_dataset, num_samples=bo_args.initinal_sequences)
    # Query Oracle for y
    initial_dataset_reward = oracle(
        initial_dataset["text"], batch_size=training_args.per_device_train_batch_size)
    initial_dataset = initial_dataset.remove_columns("reward").add_column(
        "reward", initial_dataset_reward).cast(initial_dataset.features)

    # Random choose sequences with reward < 2.0 as inital sequence
    initial_sequences = random_sampling(
        testing_dataset, num_samples=bo_args.n_sequences, constrained_reward=2.0)
    # Query Oracle for y
    initial_sequences_reward = oracle(
        initial_sequences["text"], batch_size=training_args.per_device_train_batch_size)
    initial_sequences = initial_sequences.remove_columns("reward").add_column(
        "reward", initial_sequences_reward).cast(initial_sequences.features)

    # Merge initial_sequences to initial_dataset
    initial_dataset = concatenate_datasets(
        [initial_dataset, initial_sequences])
    oracle.unload()

    buffer = {
        "dataset": initial_dataset,
        "x": [initial_sequences["text"]],
        "y": [initial_sequences["reward"]]
    }

    actor = Actor(bo_args, policy_model_args, policy_finetuning_args,
                  data_args, training_args, generating_args)

    # Startign BO loop
    for i in range(bo_args.algo_n_iterations):
        # Warming up reward models
        world_model.load()
        world_model.train(
            dataset=buffer["dataset"],
            training_args=training_args,
            data_args=data_args,
            callbacks=callbacks,
            iteration=i
        )

        # Adjusting the lookahead steps
        if actor.algo_lookahead_steps > 1 and (
            bo_args.algo_n_iterations - i < actor.algo_lookahead_steps
        ):
            actor.algo_lookahead_steps -= 1

        # Rollout training dataset for policy
        rollout_dataset = actor.rollout(
            world_model,
            buffer["x"][-1],
            n_sequences=bo_args.rollout_sequences
        )

        # Unload LLM in world model
        world_model.unload()

        # Train new policy with rolled out dataset
        actor.load_policy(iteration=i)
        actor.train_policy(dataset=rollout_dataset, data_args=data_args)
        actor.unload_policy()

        # Get the next X
        server_process = actor.load_policy_inference()
        world_model.load()
        iter_start_time = time.time()

        next_X = actor.query(
            prevX=buffer["x"],
            prevY=buffer["y"],
            reward_model=world_model,
            n_restarts=bo_args.n_restarts
        )

        iter_end_time = time.time()
        world_model.unload()
        actor.unload_policy_inference(server_process)
        print(f"Iteration {i} took {iter_end_time - iter_start_time} seconds")

        # Query Oracle for y
        oracle.load()
        next_y = oracle(
            next_X, batch_size=training_args.per_device_train_batch_size)
        oracle.unload()

        buffer["x"].append(next_X)
        buffer["y"].append(next_y)
        print("Next X", next_X)
        print("Next y", next_y)

        # Merge dataset for world_model
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
