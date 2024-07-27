import os
import json
import torch
import joblib
import pickle
import random
import psutil
import subprocess
import numpy as np
from tqdm import tqdm
from functools import partial
from datasets import Dataset, Features, load_dataset
from typing import Dict, Sequence, Tuple, Union
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from llmtuner.extras.constants import DATA_CONFIG


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


def load_bayesian_ridge_model(ckpt):
    return joblib.load(ckpt)


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


def tokenize_dataset(
    examples,
    tokenizer,
    data_args
):
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "rewards": []
    }

    for i in range(len(examples["text"])):
        input_ids = tokenizer.encode(
            examples["text"][i], add_special_tokens=False,
            # padding='max_length', truncation=True,
            # max_length=data_args.cutoff_len
        )
        attention_mask = [1] * len(input_ids)
        labels = examples["reward"][i]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["rewards"].append(labels)

    return model_inputs


def compute_regression_metrics(eval_preds: Sequence[Union[np.array, Tuple[np.array]]]) -> Dict[str, float]:
    preds, labels = eval_preds
    labels = labels.reshape(-1, 1)

    return {"mae": mean_absolute_error(labels, preds),
            "r2": r2_score(labels, preds),
            "rmse": root_mean_squared_error(labels, preds)}


def get_dataset_embedding(dataset, model, tokenizer, data_args):

    preprocess_func = partial(
        tokenize_dataset, tokenizer=tokenizer, data_args=data_args
    )
    kwargs = dict(
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=(not data_args.overwrite_cache),
        desc="Running tokenizer on dataset",
    )
    column_names = list(next(iter(dataset)).keys())
    tokenized_dataset = dataset.map(preprocess_func, batched=True,
                                    remove_columns=column_names, **kwargs)

    model_inputs = {
        "inputs_embeds": [],
        "rewards": []
    }

    for example in tqdm(tokenized_dataset):
        input_ids = torch.tensor(
            [example['input_ids']], device=model.model.device).long()
        attention_mask = torch.tensor(
            [example['attention_mask']], device=model.model.device).long()
        embeds = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if tokenizer.pad_token_id is None:
            end_index = -1
        else:
            end_index = (input_ids[0] !=
                         tokenizer.pad_token_id).nonzero()[-1].cpu()

        model_inputs['inputs_embeds'].append(
            embeds.last_hidden_state[0][end_index].squeeze().detach().cpu().tolist())
        model_inputs['rewards'].append(example['rewards'])

    embeded_dataset = Dataset.from_dict(model_inputs)
    return embeded_dataset


def random_sampling(dataset, num_samples, *args, **kwargs):
    if "constrained_reward" in kwargs:
        dataset = dataset.filter(
            lambda sample: sample["reward"] < kwargs["constrained_reward"])
    total_samples = len(dataset)
    indices = random.sample(range(total_samples), num_samples)
    return dataset.select(indices)


def run_server(cmd_string):
    try:
        server_process = subprocess.Popen(cmd_string, shell=True)
        return server_process
    except Exception as e:
        print(f"Error starting server: {e}")
        return None


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def shutdown_server(process):
    try:
        kill(process.pid)
        # process.terminate()
        print("Server shutdown successfully.")
    except Exception as e:
        print(f"Error shutting down server: {e}")


def start_process(command):
    os.system(command)


def kill_process(command):
    find_pid_command = f"""pgrep -af "{command}" """
    pid_output = subprocess.check_output(find_pid_command, shell=True)
    pid_lines = pid_output.decode().splitlines()
    pids = [line.split()[0] for line in pid_lines]

    print("PID(s) of the process:")
    print(pids)

    if pids:
        kill_pid_command = f"kill -9 {' '.join(pids)}"
        subprocess.run(kill_pid_command, shell=True)
        print("Process(es) killed.")
    else:
        print("No matching process found.")
