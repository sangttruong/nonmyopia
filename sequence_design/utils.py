import gc
import json
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

from configs import LOOKAHEAD_PROMPT, SYSTEM_PROMPT
from datasets import Dataset
from envs.proteins import PROTEINS
from envs.synthetic_fns import F1, F2
from Levenshtein import distance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from tqdm import tqdm
from transformers.utils import (
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)


def import_protein_env(mutant_ver):
    global ALLOWED_POS
    global ALLOWED_TOKENS
    global INIT_SEQ
    global MAX_SEQ
    global POLICY_PROMPT

    protein_info = PROTEINS[mutant_ver]
    POLICY_PROMPT = protein_info["POLICY_PROMPT"]
    INIT_SEQ = protein_info["INIT_SEQ"]
    MAX_SEQ = protein_info["MAX_SEQ"]
    ALLOWED_POS = protein_info["ALLOWED_POS"]
    ALLOWED_TOKENS = protein_info["ALLOWED_TOKENS"]
    return POLICY_PROMPT, INIT_SEQ, MAX_SEQ, ALLOWED_POS, ALLOWED_TOKENS


def compute_ed(original, list_str):
    return [distance(original, x) for x in list_str]


def observe_value(oracle, embs, eds, fn_ver):
    if fn_ver == "v1":
        F = F1
    elif fn_ver == "v2":
        F = F2
    outputs = oracle.predict(embs).tolist()
    outputs = [x / 5 + F(ed) for x, ed in zip(outputs, eds)]
    return outputs


def torch_gc() -> None:
    r"""
    Collects GPU or NPU memory.
    """
    gc.collect()
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_cuda_available():
        torch.cuda.empty_cache()


def get_embedding_from_server(
    server_url: str, list_sequences: List[str]
) -> List[torch.Tensor]:
    r"""
    Gets reward scores from the API server.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"messages": list_sequences}
    response = requests.post(
        server_url,
        json=payload,
        headers=headers,
        timeout=300,
    )
    embeddings = json.loads(response.text)["embedding"]
    return [embedding for embedding in embeddings]


def random_mutation(sequence, diff_prob=0.5):
    is_diff = random.choices([0, 1], weights=[1 - diff_prob, diff_prob], k=1)[0]

    if not is_diff:
        return sequence

    # Find different with INIT_SEQ
    diff_pos = []
    for i in range(len(INIT_SEQ)):
        if INIT_SEQ[i] != sequence[i]:
            diff_pos.append(i)

    possible_pos = list(set(ALLOWED_POS) - set(diff_pos))
    if len(possible_pos) == 0:
        return sequence
    edit_idx = random.choice(possible_pos)
    current_token = sequence[edit_idx]

    # Find possible tokens to edit
    possible_tokens = ALLOWED_TOKENS.copy()
    possible_tokens.remove(current_token)

    # Edit token
    new_token = random.choice(possible_tokens)
    new_sequence = sequence[:edit_idx] + new_token + sequence[edit_idx + 1 :]
    return new_sequence


def find_diff_index(a, b):
    if len(a) != len(b):
        return -1  # Return -1 if strings are not of equal length

    for i in range(len(a)):
        if a[i] != b[i]:
            return i  # Return the index of the differing character
    return -1  # Return -1 if no difference is found


def verify_seq(prev_seq, curr_seq):
    if distance(prev_seq, curr_seq) > 1:
        return 0  # Reject - Edit distance > 1

    diff_idx = find_diff_index(prev_seq, curr_seq)
    if not diff_idx:
        return 1  # Accept - Remain the same

    # Find different with INIT_SEQ
    diff_pos = []
    for i in range(len(INIT_SEQ)):
        if INIT_SEQ[i] != prev_seq[i]:
            diff_pos.append(i)

    avai_pos = list(set(ALLOWED_POS) - set(diff_pos))
    if diff_idx not in avai_pos:
        return 0  # Reject - Edit at wrong location

    possible_tokens = ALLOWED_TOKENS.copy()
    possible_tokens.remove(prev_seq[diff_idx])

    if curr_seq[diff_idx] not in possible_tokens:
        return 0  # Reject - Wrong token

    return 1  # Accept


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


def create_lookahead_sequences(args, tokenizer, oracle, ds, fn_ver):
    prevX = [[] for _ in range(args.algo_lookahead_steps + 2)]
    prevY = [[] for _ in range(args.algo_lookahead_steps + 2)]
    for _ in tqdm(range(100), desc="Creating SFT dataset"):
        for seq in tqdm(ds):
            prevX[0].append(seq["text"])
            prevY[0].append(seq["reward"])

            for las in range(args.algo_lookahead_steps + 1):
                currentX = prevX[las][-1]

                # Random mutation 10 times
                mutated_sequences = [
                    random_mutation(currentX, diff_prob=0.94) for i in range(1)
                ]
                mutation_emb = get_embedding_from_server(
                    server_url=args.embedding_model, list_sequences=mutated_sequences
                )

                mutation_y = torch.tensor(
                    observe_value(
                        oracle,
                        torch.tensor(mutation_emb),
                        compute_ed(INIT_SEQ, mutated_sequences),
                        fn_ver,
                    )
                )

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
    if "is_init" in kwargs:
        if kwargs["is_init"]:
            idx = find_idx_in_dataset(dataset, "text", MAX_SEQ)
            indices.extend([idx] * 1)
    return dataset.select(indices)


def find_idx_in_dataset(D, field, value):
    for i, s in enumerate(D):
        if s[field] == value:
            return i


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
