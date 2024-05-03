import os
import gc
import time
import torch
import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from datasets import Dataset, DatasetDict

from src.llmtuner.hparams import get_bo_args
from src.llmtuner.extras.callbacks import LogCallback
from src.llmtuner.train.tuner import export_model

from src.llmtuner.data.loader import load_single_dataset
from src.llmtuner.model import load_model, load_tokenizer
from src.llmtuner.data.parser import get_dataset_list
from src.llmtuner.data.utils import checksum, merge_dataset

from world_model import WorldModel
from utils import (
    get_dataset_embedding,
    fix_oracle_model_args,
    fix_policy_model_args,
    fix_wm_model_args,
    fix_finetuning_policy_args,
    fix_finetuning_wm_args,
)

def save_to_pkl(data, name):
    pklFile = open(name, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()

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


    emb_testing_dataset = get_dataset_embedding(testing_dataset, world_model.model, world_model.tokenizer, data_args)
    save_to_pkl(emb_testing_dataset.data, f"data/{data_args.dataset.replace('/', '_')}-{wm_model_args.wm_model_name_or_path.split('/')[-1]}-embedding-test.pkl")
    
    emb_training_dataset = get_dataset_embedding(training_dataset, world_model.model, world_model.tokenizer, data_args)
    save_to_pkl(emb_training_dataset.data, f"data/{data_args.dataset.replace('/', '_')}-{wm_model_args.wm_model_name_or_path.split('/')[-1]}-embedding-train.pkl")

    full_ds = DatasetDict({"train": emb_training_dataset, "validation": emb_testing_dataset})
    full_ds.push_to_hub(wm_model_args.export_hub_model_id,
                        token=wm_model_args.hf_hub_token,
                        commit_message="Upload data")
    
if __name__ == '__main__':
    main()
