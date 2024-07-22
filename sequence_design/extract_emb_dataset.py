from typing import Any, Dict, Optional
from llmtuner.hparams import get_train_args
from llmtuner.model import load_model, load_tokenizer
from llmtuner.data.parser import get_dataset_list
from llmtuner.data.utils import merge_dataset
from utils import custom_load_dataset
from datasets import DatasetDict
from embed_text_package import embed_text
from tqdm import tqdm


def main(args: Optional[Dict[str, Any]] = None, callbacks=None):
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        args)

    # Load (pre-trained) Tokenizer (according to model)
    # and model via Llama-Factory functions
    tokenizer = load_tokenizer(model_args)
    model = load_model(
        tokenizer,
        model_args,
        finetuning_args,
        is_trainable=False,
        add_valuehead=False
    )

    all_data_splits = ["train", "validation"]
    new_dataset = DatasetDict()
    # Loop for training and testing-datasets.
    for data_split in all_data_splits:
        # load the dataset into dataset_split
        with training_args.main_process_first(desc="load training dataset"):
            all_datasets = []
            data_args.split = data_split
            for dataset_attr in get_dataset_list(data_args):
                dataset = custom_load_dataset(
                    dataset_attr, data_args, model_args)
                all_datasets.append(dataset)
            dataset_split = merge_dataset(
                all_datasets, data_args, training_args)

        # create batch-structure:
        batch_size = 64  # This parameter should be added in command
        batches_sentences = []
        for i in tqdm(range(0, len(dataset_split), batch_size)):
            batches_sentences.append(dataset_split['text'][i:i+batch_size])

        # get embeddings
        emb = embed_text.get_embeddings(batches_sentences, model, tokenizer)

        # Unpack batches and add to dataset
        emb_all = [item for sublist in emb for item in sublist]
        print("Adding embeddings into new dataset. \
              This might take a few minutes.")
        new_dataset[data_split] = dataset_split.add_column(name='embeddings',
                                                           column=emb_all)

    new_dataset.push_to_hub(model_args.export_hub_model_id,
                            token=model_args.hf_hub_token,
                            commit_message="Upload data")


if __name__ == '__main__':
    main()
