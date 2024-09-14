import os
import pickle
import random
import subprocess
import time
from dataclasses import field, make_dataclass
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import psutil
import requests
import torch
import yaml
from configs import ALLOWED_TOKENS, LOOKAHEAD_PROMPT, POLICY_PROMPT, SYSTEM_PROMPT
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm


def random_mutation(sequence):
    sequence_length = len(sequence)
    edit_idxs = random.choice(list(range(sequence_length)))
    edit_tokens = random.choice(ALLOWED_TOKENS)
    new_sequence = sequence[:edit_idxs] + edit_tokens + sequence[edit_idxs + 1 :]
    return new_sequence


def format_prompt(tokenizer, prevX: List[List[str]], prevY: List[List[float]]):
    # prevX, prevY: n_steps x n_proteins
    n_steps = len(prevX)
    n_proteins = len(prevX[0])

    prompts = []
    for pi in range(n_proteins):
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        for sid in range(n_steps):
            X = prevX[sid][pi]
            y = prevY[sid][pi]

            if sid == 0:
                prompt.append(
                    {
                        "role": "user",
                        "content": POLICY_PROMPT.format(protein=X)
                        + LOOKAHEAD_PROMPT.format(reward=y),
                    }
                )
            else:
                prompt.append({"role": "assistant", "content": X})
                prompt.append(
                    {"role": "user", "content": LOOKAHEAD_PROMPT.format(reward=y)}
                )

        prompt = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt)

    return prompts


def create_lookahead_sequences(args, tokenizer, embedder, oracle, ds):
    prevX = [[] for _ in range(args.algo_lookahead_steps + 2)]
    prevY = [[] for _ in range(args.algo_lookahead_steps + 2)]
    for seq in tqdm(ds):
        prevX[0].append(seq["text"])
        prevY[0].append(seq["reward"])

        for las in range(args.algo_lookahead_steps + 1):
            currentX = prevX[las][-1]

            # Random mutation 10 times
            mutated_sequences = [random_mutation(currentX) for i in range(10)]
            mutation_ds = Dataset.from_dict({"text": mutated_sequences})
            mutation_emb = (
                embedder.get_embeddings(
                    DataLoader(mutation_ds, batch_size=1),
                    args.embedding_model_name_or_path,
                    ["text"],
                )
                .data["text"]
                .to_pylist()
            )
            mutation_y = oracle.predict(torch.tensor(mutation_emb))

            # Select the best
            best_idx = mutation_y.argmax()
            prevX[las + 1].append(mutated_sequences[best_idx])
            prevY[las + 1].append(mutation_y[best_idx])

    list_text = format_prompt(tokenizer, prevX, prevY)

    return Dataset.from_dict({"text": list_text})


def create_dataclass_from_dict(class_name: str, data: dict):
    """
    Function to create a dataclass dynamically from a dictionary
    """
    fields = [(key, type(value), field(default=value)) for key, value in data.items()]
    return make_dataclass(class_name, fields)


def read_yaml_to_dynamic_dataclass(file_path: str, class_name: str = "DynamicConfig"):
    """
    Function to read YAML file and create a dataclass
    """
    with open(file_path, "r", encoding="utf8") as file:
        data = yaml.safe_load(file)
    return create_dataclass_from_dict(class_name, data)


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


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


def compute_regression_metrics(
    eval_preds: Sequence[Union[np.array, Tuple[np.array]]]
) -> Dict[str, float]:
    preds, labels = eval_preds
    labels = labels.reshape(-1, 1)

    return {
        "mae": mean_absolute_error(labels, preds),
        "r2": r2_score(labels, preds),
        "rmse": root_mean_squared_error(labels, preds),
    }


def random_sampling(dataset, num_samples, *args, **kwargs):
    if "constrained_reward" in kwargs:
        dataset = dataset.filter(
            lambda sample: sample["reward"] < kwargs["constrained_reward"]
        )
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


def check_health(url):
    server_ok = False
    while server_ok is False:
        try:
            # Send a GET request to the health check endpoint
            response = requests.get(url)

            # Check if the server is healthy
            if response.status_code == 200:
                server_ok = True
            else:
                time.sleep(1)

        except requests.exceptions.RequestException as e:
            time.sleep(1)
    return server_ok
