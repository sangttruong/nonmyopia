from typing import Any, Dict, Optional
from datasets import DatasetDict
from llmtuner.hparams import get_train_args
from llmtuner.data.parser import get_dataset_list
from llmtuner.data.utils import merge_dataset
from llmtuner.model import load_model, load_tokenizer
from utils import custom_load_dataset, get_dataset_embedding, save_to_pkl


def main(args: Optional[Dict[str, Any]] = None, callbacks=None):
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        args)

    tokenizer = load_tokenizer(model_args)
    model = load_model(
        tokenizer,
        model_args,
        finetuning_args,
        is_trainable=False,
        add_valuehead=False
    )

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
        testing_dataset, model, tokenizer, data_args)
    # save_to_pkl(emb_testing_dataset.data,
    #             f"data/{data_args.dataset.replace('/', '_')}-"
    #             f"{model_args.model_name_or_path.split('/')[-1]}-"
    #             "embedding-test.pkl")

    emb_training_dataset = get_dataset_embedding(
        training_dataset, model, tokenizer, data_args)
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
