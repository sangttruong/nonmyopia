import copy
import os
from datetime import datetime

import torch
import yaml
from datasets import Dataset
from peft import AutoPeftModelForCausalLM

from policy import Policy
from torch.utils.data import DataLoader
from utils import run_server, shutdown_server, start_process


def collate_fn(data):
    zipped = zip(data)
    return list(zipped)


class Actor:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.policy = Policy(self.config)
        self.algo_lookahead_steps = config.algo_lookahead_steps

        if self.config.algo not in ["HES-TS-AM", "qEI", "qSR", "qUCB", "qPI", "qKG"]:
            raise NotImplementedError(
                f"Acquisition function `{self.config.algo}` is not implemented!"
            )

    def query(self, prevX, prevY, embedder, reward_model, n_restarts=3):
        # Query the next sequence
        X = prevX[-1]
        n_sequences = len(X)
        # >>> n_sequences

        if self.config.algo == "HES-TS-AM":
            X_returned = []
            rewards = []
            for rid in range(n_restarts):
                local_prevX = copy.deepcopy(prevX)
                local_prevy = copy.deepcopy(prevY)
                local_X = local_prevX[-1]

                for step in range(self.algo_lookahead_steps):
                    next_X = self.policy.generate(local_X, local_prevX, local_prevy)
                    next_X_ds = Dataset.from_dict({"text": next_X})
                    next_X_ds_emb = (
                        embedder.get_embeddings(
                            DataLoader(next_X_ds, batch_size=1),
                            self.config.embedding_model_name_or_path,
                            ["text"],
                        )
                        .data["text"]
                        .to_pylist()
                    )

                    next_y = (
                        reward_model.sample(
                            torch.tensor(next_X_ds_emb),
                            sample_size=self.config.sample_size,
                        )
                        .mean(0)
                        .float()
                        .detach()
                        .cpu()
                        .tolist()
                    )

                    local_prevX.append(next_X)
                    local_prevy.append(next_y)
                    local_X = next_X

                action_X = self.policy.generate(local_X, local_prevX, local_prevy)
                action_X_ds = Dataset.from_dict({"text": action_X})
                action_X_ds_emb = (
                    embedder.get_embeddings(
                        DataLoader(action_X_ds, batch_size=1),
                        self.config.embedding_model_name_or_path,
                        ["text"],
                    )
                    .data["text"]
                    .to_pylist()
                )
                action_y = (
                    reward_model.sample(torch.tensor(action_X_ds_emb))
                    .mean(0)
                    .float()
                    .detach()
                    .cpu()
                    .tolist()
                )

                X_returned.append(local_prevX)
                rewards.append(action_y)

            # For each sequence, find the best next sequence across n_restarts based on computed reward
            best_idx = torch.tensor(rewards).argmax(dim=0).numpy().tolist()
            output = []

            for bi, si in zip(best_idx, list(range(n_sequences))):
                output.append(X_returned[bi][0][si])

            return output

        else:
            raise NotImplementedError

    def format_query(self, dataset):
        def format_query_fn(queries):
            formatted_query = []
            for query in queries["prompt"]:
                formatted_query.append(
                    self.policy.tokenizer.apply_chat_template(query, tokenize=False)
                )
            return {"text": formatted_query}

        # Dataset must have a column 'text'
        dataset = dataset.map(format_query_fn, batched=True, remove_columns=["prompt"])
        return dataset

    def create_dataset(
        self,
        prevX,
        prevY,
        sequence_length=237,
    ):
        # Create dataset
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [{"role": "user", "content": seq}]
                    for seq in self.policy.format_prompt(
                        prevX=prevX,
                        prevY=prevY,
                    )
                ],
            }
        )

        return self.format_query(dataset)

    def train_policy(self, iteration, dataset) -> None:
        timestamp = datetime.today().isoformat()
        algo = self.config.algo
        if "HES" in algo:
            algo = "qMultiStepHEntropySearch"

        # Preprocess queries to LLM's format
        dataset_name = f"ppo_{timestamp}"
        dataset.to_csv(f"data/{dataset_name}/{dataset_name}.csv")

        # Start PPO training
        # Edit the checkpoint
        output_dir = os.path.join(self.config.output_dir, f"{iteration}")
        with open("configs/ppo.yaml", "r", encoding="utf8") as stream:
            loaded_configs = yaml.safe_load(stream)
            loaded_configs["output_dir"] = output_dir
            if iteration == 0:
                loaded_configs["model_name_or_path"] = (
                    self.config.policy_model_name_or_path
                )
            else:
                loaded_configs["model_name_or_path"] = os.path.join(
                    self.config.output_dir, f"{iteration-1}"
                )
            loaded_configs["query_dataset"] = f"data/{dataset_name}"

        training_config_file = f"configs/ppo_{timestamp}.yaml"
        with open(training_config_file, "w", encoding="utf8") as stream:
            yaml.dump(
                loaded_configs, stream, default_flow_style=False, allow_unicode=True
            )

        # Start reward server
        server_process = run_server(
            f"python -m llmppo.reward_server --model acqfs.{algo} --config {training_config_file}"
        )

        # Start PPO training process
        start_process(
            f"CUDA_VISIBLE_DEVICES={self.config.ppo_gpu} accelerate launch "
            "--config_file configs/single_config.yaml "
            f"-m llmppo.ppo_vllm --config {training_config_file}"
        )

        # Stop reward server
        shutdown_server(server_process)

        # Remove temp config file
        os.system(f"rm -rf {training_config_file}")

        # Merge Lora adapter
        if loaded_configs["use_peft"]:
            model = AutoPeftModelForCausalLM.from_pretrained(output_dir)
            model = model.merge_and_unload().to(torch.bfloat16)
            model.save_pretrained(output_dir)
