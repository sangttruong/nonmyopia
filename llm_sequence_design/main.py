import os
import time
import torch
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
from configs import (
    initinal_samples,
    samples_per_iteration,
    number_of_iterations,
    TRAINING_DATA_BUFFER,
    TESTING_DATA_BUFFER,
)
from utils import (
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

    # Fixing args
    fix_wm_model_args(wm_model_args)
    fix_oracle_model_args(oracle_model_args)
    fix_policy_model_args(policy_model_args)
    fix_finetuning_wm_args(wm_finetuning_args)
    fix_finetuning_policy_args(policy_finetuning_args)

    # Initializing models
    # oracle = Oracle(oracle_model_args, wm_finetuning_args)
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

    with training_args.main_process_first(desc="load testing dataset"):
        all_datasets = []
        data_args.split = "validation"
        for dataset_attr in get_dataset_list(data_args):
            all_datasets.append(load_single_dataset(
                dataset_attr, policy_model_args, data_args))
        testing_dataset = merge_dataset(
            all_datasets, data_args, training_args)

    # Initializing training buffer
    # Randomly sample from training dataset
    initial_dataset = random_sampling(
        training_dataset, num_samples=initinal_samples)
    buffer = {
        "dataset": initial_dataset,
        "eval_rmse": []
    }
    actor = Actor(bo_args, policy_model_args,
                  policy_finetuning_args, training_args, generating_args)

    # Startign BO loop
    for i in range(number_of_iterations):
        # Warming up reward models
        world_model.train_v2(
            dataset=buffer["dataset"],
            training_args=training_args,
            data_args=data_args,
            callbacks=callbacks
        )

        # Adjusting the lookahead steps
        if actor.algo_lookahead_steps > 1 and (
            number_of_iterations - i < actor.algo_lookahead_steps
        ):
            actor.algo_lookahead_steps -= 1

        # Get the next action
        iter_start_time = time.time()
        next_x = actor.query(buffer["dataset"], world_model)

        iter_end_time = time.time()
        print(f"Iteration {i} took {iter_end_time - iter_start_time} seconds")

        # Final evaluation

        # Exporting the best model


if __name__ == '__main__':
    main()
