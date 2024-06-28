import os
import json
import pickle
from functools import partial
from typing import Any, Dict, Optional
from datasets import DatasetDict, load_dataset, Features
from llmtuner.hparams import get_train_args
from llmtuner.data.parser import get_dataset_list
from llmtuner.data.utils import merge_dataset
from llmtuner.extras.constants import DATA_CONFIG
from surr_model import SurrModel
from utils import get_dataset_embedding


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


def custom_load_dataset(dataset_attr, data_args, model_args):
    dataset = load_dataset(
        path=dataset_attr.dataset_name,
        name=dataset_attr.subset,
        data_dir=dataset_attr.folder,
        split=data_args.split,
        cache_dir=model_args.cache_dir,
        token=model_args.hf_hub_token
    )
    dataset = dataset.select(range(200))
    column_names = list(next(iter(dataset)).keys())
    features = Features.from_dict(
        {
            "text": {"dtype": "string", "_type": "Value"},
            "reward": {"dtype": "float", "_type": "Value"}
        }
    )
    dataset_info = get_data_info(data_args)
    for column_name in ["text", "reward"]:
        dataset_attr.set_attr(
            column_name, dataset_info[dataset_attr.dataset_name]["columns"])

    convert_func = partial(convert_oracle, dataset_attr=dataset_attr)
    dataset = dataset.map(
        convert_func,
        batched=True,
        remove_columns=column_names,
        features=features
    )
    return dataset


def main(args: Optional[Dict[str, Any]] = None, callbacks=None):
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        args)

    surr_model = SurrModel(model_args, finetuning_args)
    surr_model.load()

    # Initializing full dataset
    with training_args.main_process_first(desc="load training dataset"):
        all_datasets = []
        data_args.split = "train"
        for dataset_attr in get_dataset_list(data_args):
            dataset = custom_load_dataset(
                dataset_attr, data_args, model_args)
            all_datasets.append(dataset)
        training_dataset = merge_dataset(
            all_datasets, data_args, training_args)

    with training_args.main_process_first(desc="load testing dataset"):
        all_datasets = []
        data_args.split = "validation"
        for dataset_attr in get_dataset_list(data_args):
            dataset = custom_load_dataset(
                dataset_attr, data_args, model_args)
            all_datasets.append(dataset)
        testing_dataset = merge_dataset(
            all_datasets, data_args, training_args)

    emb_testing_dataset = get_dataset_embedding(
        testing_dataset, surr_model.model, surr_model.tokenizer, data_args)
    # save_to_pkl(emb_testing_dataset.data,
    #             f"data/{data_args.dataset.replace('/', '_')}-"
    #             f"{model_args.model_name_or_path.split('/')[-1]}-"
    #             "embedding-test.pkl")

    emb_training_dataset = get_dataset_embedding(
        training_dataset, surr_model.model, surr_model.tokenizer, data_args)
    # save_to_pkl(emb_training_dataset.data,
    #             f"data/{data_args.dataset.replace('/', '_')}-"
    #             f"{model_args.model_name_or_path.split('/')[-1]}-"
    #             "embedding-train.pkl")

    full_ds = DatasetDict({"train": emb_training_dataset,
                          "validation": emb_testing_dataset})
    full_ds.push_to_hub(model_args.export_hub_model_id,
                        token=model_args.hf_hub_token,
                        commit_message="Upload data")


if __name__ == '__main__':
    main()
