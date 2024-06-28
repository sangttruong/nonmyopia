import os
import gc
import time
import torch
import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from datasets import Dataset, DatasetDict
from datasets import load_dataset
import json

from src.llmtuner.hparams import get_bo_args

from llmtuner.data.parser import get_dataset_list
from llmtuner.data.utils import merge_dataset
from llmtuner.extras.constants import DATA_CONFIG

from world_model import WorldModel
from utils import (
    get_dataset_embedding,
    fix_oracle_model_args,
    fix_policy_model_args,
    fix_wm_model_args,
    fix_finetuning_policy_args,
    fix_finetuning_wm_args,
)
from functools import partial
from datasets import Dataset, DatasetDict, load_dataset, Features


def convert_oracle(examples, dataset_attr):
    outputs = {"text": [], "reward": []}
    for i, messages in enumerate(examples[dataset_attr.text]):
        outputs["text"].append(messages)
        outputs["reward"].append(float(examples[dataset_attr.reward][i]))
    return outputs


def save_to_pkl(data, name):
    pklFile = open(name, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()


def get_data_info(data_args):
    with open(os.path.join(data_args.dataset_dir, DATA_CONFIG), "r") as f:
        dataset_info = json.load(f)
    return dataset_info


def main(
    args: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    (
        wm_model_args,
        oracle_model_args,
        policy_model_args,
        data_args,
        training_args,
        wm_finetuning_args,
        policy_finetuning_args,
        generating_args,
        bo_args,
    ) = get_bo_args(args)

    # Fixing args
    fix_wm_model_args(wm_model_args)
    fix_oracle_model_args(oracle_model_args)
    fix_policy_model_args(policy_model_args)
    fix_finetuning_wm_args(wm_finetuning_args)
    fix_finetuning_policy_args(policy_finetuning_args)

    world_model = WorldModel(wm_model_args, wm_finetuning_args)
    world_model.load()

    # Initializing full dataset
    with training_args.main_process_first(desc="load training dataset"):
        all_datasets = []
        data_args.split = "train"
        for dataset_attr in get_dataset_list(data_args):
            dataset = load_dataset(
                path=dataset_attr.dataset_name,
                name=dataset_attr.subset,
                data_dir=dataset_attr.folder,
                split=data_args.split,
                cache_dir=wm_model_args.cache_dir,
                token=wm_model_args.hf_hub_token,
            )
            column_names = list(next(iter(dataset)).keys())
            features = Features.from_dict(
                {
                    "text": {"dtype": "string", "_type": "Value"},
                    "reward": {"dtype": "float", "_type": "Value"},
                }
            )
            dataset_info = get_data_info(data_args)
            for column_name in ["text", "reward"]:
                dataset_attr.set_attr(
                    column_name, dataset_info[dataset_attr.dataset_name]["columns"])
            convert_func = partial(convert_oracle, dataset_attr=dataset_attr)

            dataset.map(
                convert_func,
                batched=True,
                remove_columns=column_names,
                features=features,
            )

            all_datasets.append(dataset)
        training_dataset = merge_dataset(all_datasets, data_args, training_args)

    with training_args.main_process_first(desc="load testing dataset"):
        all_datasets = []
        data_args.split = "validation"
        for dataset_attr in get_dataset_list(data_args):
            dataset = load_dataset(
                path=dataset_attr.dataset_name,
                name=dataset_attr.subset,
                data_dir=dataset_attr.folder,
                split=data_args.split,
                cache_dir=wm_model_args.cache_dir,
                token=wm_model_args.hf_hub_token,
            )

            column_names = list(next(iter(dataset)).keys())
            features = Features.from_dict(
                {
                    "text": {"dtype": "string", "_type": "Value"},
                    "reward": {"dtype": "float", "_type": "Value"},
                }
            )
            dataset_info = get_data_info(data_args)
            for column_name in ["text", "reward"]:
                dataset_attr.set_attr(
                    column_name, dataset_info[dataset_attr.dataset_name]["columns"])
            convert_func = partial(convert_oracle, dataset_attr=dataset_attr)

            dataset.map(
                convert_func,
                batched=True,
                remove_columns=column_names,
                features=features,
            )
            all_datasets.append(dataset)

        testing_dataset = merge_dataset(all_datasets, data_args, training_args)

    emb_testing_dataset = get_dataset_embedding(
        testing_dataset, world_model.model, world_model.tokenizer, data_args, 
    )
    save_to_pkl(
        emb_testing_dataset.data,
        f"data/{data_args.dataset.replace('/', '_')}-{wm_model_args.wm_model_name_or_path.split('/')[-1]}-embedding-test.pkl",
    )

    emb_training_dataset = get_dataset_embedding(
        training_dataset, world_model.model, world_model.tokenizer, data_args
    )
    save_to_pkl(
        emb_training_dataset.data,
        f"data/{data_args.dataset.replace('/', '_')}-{wm_model_args.wm_model_name_or_path.split('/')[-1]}-embedding-train.pkl",
    )

    full_ds = DatasetDict(
        {"train": emb_training_dataset, "validation": emb_testing_dataset}
    )
    full_ds.push_to_hub(
        wm_model_args.export_hub_model_id,
        token=wm_model_args.hf_hub_token,
        commit_message="Upload data",
    )


if __name__ == "__main__":
    main()
