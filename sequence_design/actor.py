import copy
import os
import pickle
from datetime import datetime

import numpy as np
import torch
import yaml
from botorch.acquisition import (
    qExpectedImprovement,
    qKnowledgeGradient,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from configs import TEMPLATED_LOOKAHEAD_PROMPT
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from policy import Policy
from torch.utils.data import DataLoader
from utils import (
    check_health,
    format_prompt,
    get_embedding_from_server,
    run_server,
    shutdown_server,
    start_process,
)


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

    def query(self, iteration, prevX, prevY, reward_model, n_restarts=3):
        # Query the next sequence
        X = prevX[-1]
        n_sequences = len(X)
        # >>> n_sequences

        X_returned = []
        rewards = []
        for rid in range(n_restarts):
            local_prevX = copy.deepcopy(prevX)
            local_prevy = copy.deepcopy(prevY)

            for step in range(self.algo_lookahead_steps):
                next_X = self.policy.generate(local_prevX, local_prevy)
                next_X_ds_emb = get_embedding_from_server(
                    server_url=self.config.embedding_model, list_sequences=next_X
                )

                next_y = (
                    reward_model.sample(
                        torch.tensor(next_X_ds_emb).unsqueeze(-2),
                        sample_size=self.config.sample_size,
                    )
                    .mean(0)
                    .reshape(-1)
                    .float()
                    .detach()
                    .cpu()
                    .tolist()
                )

                local_prevX.append(next_X)
                local_prevy.append(next_y)

            action_X = self.policy.generate(local_prevX, local_prevy)
            action_X_ds_emb = get_embedding_from_server(
                server_url=self.config.embedding_model, list_sequences=action_X
            )
            if self.config.algo == "HES-TS-AM":
                action_y = (
                    reward_model.sample(
                        torch.tensor(action_X_ds_emb).unsqueeze(-2),
                        sample_size=self.config.sample_size,
                    )
                    .mean(0)
                    .squeeze(-1)
                    .float()
                    .detach()
                    .cpu()
                    .tolist()
                )

            elif self.config.algo == "qSR":
                bo_acqf = qSimpleRegret(model=reward_model)
                action_y = (
                    bo_acqf(torch.tensor(action_X_ds_emb).unsqueeze(-2)).cpu().tolist()
                )

            elif self.config.algo == "qEI":
                best_f = np.array(prevY)
                best_f = np.max(best_f, axis=0)
                bo_acqf = qExpectedImprovement(
                    model=reward_model, best_f=torch.tensor(best_f)
                )
                action_y = (
                    bo_acqf(torch.tensor(action_X_ds_emb).unsqueeze(-2)).cpu().tolist()
                )

            elif self.config.algo == "qPI":
                best_f = np.array(prevY)
                best_f = np.max(best_f, axis=0)
                self.bo_acqf = qProbabilityOfImprovement(
                    model=reward_model, best_f=torch.tensor(best_f)
                )
                action_y = (
                    bo_acqf(torch.tensor(action_X_ds_emb).unsqueeze(-2)).cpu().tolist()
                )

            elif self.config.algo == "qUCB":
                bo_acqf = qUpperConfidenceBound(model=reward_model, beta=0.1)
                action_y = (
                    bo_acqf(torch.tensor(action_X_ds_emb).unsqueeze(-2)).cpu().tolist()
                )

            elif self.config.algo == "qKG":
                bo_acqf = qKnowledgeGradient(
                    model=reward_model,
                    num_fantasies=1,
                )
                action_y = (
                    bo_acqf(torch.tensor(action_X_ds_emb).unsqueeze(-2)).cpu().tolist()
                )

            local_prevX.append(action_X)
            local_prevy.append(action_y)

            X_returned.append(local_prevX)
            rewards.append(action_y)

        # For each sequence, find the best next sequence across n_restarts based on computed reward
        best_idx = torch.tensor(rewards).argmax(dim=0).numpy().tolist()
        output = []

        for bi, si in zip(best_idx, list(range(n_sequences))):
            output.append(X_returned[bi][iteration + 1][si])

        with open(f"{self.config.output_dir}/trajectory_{iteration}.pkl", "wb") as f:
            pickle.dump((best_idx, rewards, X_returned), f)

        return output

    def create_dataset(
        self,
        prevX,
        prevY,
    ):
        data_dict = {"text": []}
        data_dict.update({f"text{i+1}": [] for i in range(self.algo_lookahead_steps)})
        # Create dataset
        lookahead_prompt = None
        for model_type in TEMPLATED_LOOKAHEAD_PROMPT:
            if model_type in self.config.policy_model_name_or_path.lower():
                lookahead_prompt = TEMPLATED_LOOKAHEAD_PROMPT[model_type]
                break
        assert (
            lookahead_prompt is not None
        ), "Please define template for lookahead manually."

        for seq in format_prompt(
            tokenizer=self.policy.tokenizer,
            prevX=prevX,
            prevY=prevY,
        ):
            data_dict["text"].append(seq)
            for i in range(self.algo_lookahead_steps):
                data_dict[f"text{i+1}"].append(lookahead_prompt)

        for key, value in data_dict.items():
            data_dict[key] = value * self.config.n_restarts

        dataset = Dataset.from_dict(data_dict)
        return dataset

    def train_policy(
        self,
        iteration,
        dataset,
    ) -> None:
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
                    f"ckpts/sft_model_n{self.config.n_sequences}_lah{self.config.algo_lookahead_steps}_s{self.config.seed}"
                )
            else:
                loaded_configs["model_name_or_path"] = os.path.join(
                    self.config.output_dir, f"{iteration-1}"
                )
            loaded_configs["query_dataset"] = f"data/{dataset_name}"
            loaded_configs["reward_model"] = (
                f"http://localhost:{self.config.reward_model_port}"
            )

        training_config_file = f"configs/ppo_{timestamp}.yaml"
        with open(training_config_file, "w", encoding="utf8") as stream:
            yaml.dump(
                loaded_configs, stream, default_flow_style=False, allow_unicode=True
            )

        # Start reward server
        server_process = run_server(
            f"python -m lampo.reward_server --model acqfs.{algo} --config {training_config_file} --port {self.config.reward_model_port}"
        )

        # Waiting for reward server
        check_health(loaded_configs["reward_model"] + "/health")

        # Start PPO training process
        start_process(
            f"CUDA_VISIBLE_DEVICES={self.config.ppo_gpu} accelerate launch --main_process_port {self.config.main_process_port} "
            "--config_file configs/single_config.yaml "
            f"-m lampo.ppo_vllm --config {training_config_file}"
        )

        # Stop reward server
        shutdown_server(server_process)

        # Remove temp config file
        os.system(f"rm -rf {training_config_file}")
        os.system(f"rm -rf data/{dataset_name}")

        # Merge Lora adapter
        if loaded_configs["use_peft"]:
            model = AutoPeftModelForCausalLM.from_pretrained(output_dir)
            model = model.merge_and_unload().to(torch.bfloat16)
            model.save_pretrained(output_dir)
