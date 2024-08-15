import os
import copy
import torch
import random
import yaml
from datetime import datetime
from datasets import Dataset
from torch.utils.data import DataLoader
from peft import AutoPeftModelForCausalLM

from configs import ALLOWED_TOKENS
from policy import Policy
from utils import start_process, kill_process


def collate_fn(data):
    zipped = zip(data)
    return list(zipped)


class Actor:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.policy = Policy(self.config)
        self.config.output_dir = os.path.join(self.config.output_dir, "policy")

        self.algo_lookahead_steps = config.algo_lookahead_steps

        if self.config.algo not in ["HES-TS-AM", "qEI", "qSR"]:
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

    @torch.no_grad()
    def rollout(
        self,
        embedder,
        reward_model,
        sequences,
        n_sequences=16,
        sequence_length=237,
    ):
        if len(sequences) <= n_sequences:
            n_input_sequences = len(sequences)
            sequences = [sequences * (n_sequences // n_input_sequences)]
            n_sequences = len(sequences[0])
        else:
            sequences = [sequences[-n_sequences:]]

        # Deprecated
        # sequences = [[''.join(random.choices(ALLOWED_TOKENS, k=sequence_length)) for _ in range(n_sequences)]]

        for i in range(self.algo_lookahead_steps):
            step_sequences = []

            edit_idxs = random.choices(list(range(sequence_length)), k=n_sequences)
            edit_tokens = random.choices(ALLOWED_TOKENS, k=n_sequences)
            for sid, (idx, token) in enumerate(zip(edit_idxs, edit_tokens)):
                new_sequence = sequences[i][sid]
                new_sequence = new_sequence[:idx] + token + new_sequence[idx + 1 :]
                step_sequences.append(new_sequence)

            sequences.append(step_sequences)

        # Infer reward
        flatten_sequences = [s for ss in sequences for s in ss]
        flatten_sequences_emb = (
            embedder.get_embeddings(
                DataLoader(
                    Dataset.from_dict({"text": flatten_sequences}), batch_size=1
                ),
                embedder.which_model,
                ["text"],
            )
            .data["text"]
            .to_pylist()
        )
        flatten_sequences_emb = torch.tensor(flatten_sequences_emb)

        rewards = reward_model.sample(flatten_sequences_emb, sample_size=1)
        rewards = rewards.reshape(1, -1, n_sequences).mean(0)
        # Create dataset
        data_dict = {
            "prompt": [],
        }
        for i in range(self.algo_lookahead_steps + 1):
            data_dict["prompt"].extend(
                [
                    [{"role": "user", "content": seq}]
                    for seq in self.policy.format_prompt(
                        X=sequences[i],
                        prevX=sequences[: i + 1],
                        prevY=rewards[: i + 1].float().detach().cpu().tolist(),
                    )
                ]
            )

        return Dataset.from_dict(data_dict)

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

    def train_policy(self, iteration, dataset) -> None:
        timestamp = datetime.today().isoformat()
        algo = self.config.algo
        if "HES" in algo:
            algo = "qMultiStepHEntropySearch"

        # Preprocess queries to LLM's format
        dataset = self.format_query(dataset)
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
        start_process(
            f"python -m llmppo.reward_server --model acqfs.{algo} --config {training_config_file} &"
        )

        # Start PPO training process
        start_process(
            f"CUDA_VISIBLE_DEVICES={self.config.ppo_gpu} accelerate launch "
            "--config_file configs/single_config.yaml "
            f"-m llmppo.ppo_vllm --config {training_config_file}"
        )

        # Stop reward server
        kill_process(
            f"python -m llmppo.reward_server --model acqfs.{algo} --config {training_config_file} &"
        )

        # Merge Lora adapter
        if loaded_configs["use_peft"]:
            model = AutoPeftModelForCausalLM.from_pretrained(output_dir)
            model = model.merge_and_unload().to(torch.bfloat16)
            model.save_pretrained(output_dir)
