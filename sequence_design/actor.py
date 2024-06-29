import os
import copy
import torch
import random
from datasets import Dataset
from functools import partial
from transformers import DataCollatorWithPadding
from typing import TYPE_CHECKING, List, Optional

from llmtuner.extras.callbacks import FixValueHeadModelCallback
from llmtuner.extras.callbacks import LogCallback
from llmtuner.extras.misc import fix_valuehead_checkpoint
from llmtuner.extras.ploting import plot_loss
from llmtuner.train.utils import create_ref_model
from llmtuner.data.preprocess import preprocess_unsupervised_dataset
from llmtuner.data.template import get_template_and_fix_tokenizer
from acqfs import (
    acqf_random,
)
from configs import ALLOWED_TOKENS
from policy import Policy, PolicyPPOTrainer

def collate_fn(data):
    zipped = zip(data)
    return list(zipped)
    
class Actor:
    def __init__(self, bo_args, policy_model_args, policy_finetuning_args, data_args, training_args, generating_args):
        self.bo_args = bo_args
        self.data_args = data_args
        self.policy_model_args = policy_model_args
        self.policy_finetuning_args = policy_finetuning_args
        self.policy = Policy(self.policy_model_args, self.policy_finetuning_args, self.data_args)
        self.training_args = copy.deepcopy(training_args)
        self.training_args.output_dir = os.path.join(
            self.training_args.output_dir, "policy")

        self.generating_args = generating_args

        self.algo_lookahead_steps = bo_args.algo_lookahead_steps

        if self.bo_args.algo != "HES-TS-AM":
            raise NotImplementedError

    def load_policy(self, *args, **kwargs):
        self.policy.load(*args, **kwargs)

    def unload_policy(self):
        self.policy.unload()

    def load_policy_inference(self, *args, **kwargs):
        return self.policy.load_inference(*args, **kwargs)

    def unload_policy_inference(self, *args, **kwargs):
        return self.policy.unload_inference(*args, **kwargs)
        
    def query(self, prevX, prevY, reward_model, n_restarts=3):
        # Query the next sequence
        X = prevX[-1]
        n_sequences = len(X)
        # >>> n_sequences

        if self.bo_args.algo == "HES-TS-AM":
            X_returned = []
            rewards = []
            for rid in range(n_restarts):
                local_prevX = copy.deepcopy(prevX)
                local_prevy = copy.deepcopy(prevY)
                local_X = local_prevX[-1]
                
                for step in range(self.algo_lookahead_steps):
                    next_X = self.policy.generate(local_X, local_prevX, local_prevy, generating_args=self.generating_args)
                    next_y = reward_model.sample(next_X, batch_size=self.training_args.per_device_train_batch_size).mean(0).float().detach().cpu().tolist()

                    local_prevX.append(next_X)
                    local_prevy.append(next_y)
                    local_X = next_X

                action_X = self.policy.generate(local_X, local_prevX, local_prevy, generating_args=self.generating_args)
                action_y = reward_model.sample(action_X).mean(0).float().detach().cpu().tolist()

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
        reward_model,
        sequences,
        n_sequences = 16,
        sequence_length = 237,
    ):
        if len(sequences) <= n_sequences:
            n_input_sequences = len(sequences)
            sequences = [sequences*(n_sequences // n_input_sequences)]
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
                new_sequence = new_sequence[:idx] + token + new_sequence[idx+1:]
                step_sequences.append(new_sequence)
                
            sequences.append(step_sequences)
                
        # Infer reward
        flatten_sequences = [s for ss in sequences for s in ss ]
        rewards = reward_model.sample(flatten_sequences, sample_size=1, batch_size=self.training_args.per_device_train_batch_size)
        rewards = rewards.reshape(1, -1, n_sequences).mean(0)

        # Create dataset
        data_dict = {"prompt": [], "response": [], "reward": [], "system": [], "tools": []}
        for i in range(self.algo_lookahead_steps):
            data_dict["prompt"].extend(
                [
                    [{"role": "user", "content": seq}] for seq in \
                        self.policy.format_prompt(
                            X=sequences[i], 
                            prevX=sequences[:i+1], 
                            prevY=rewards[:i+1].float().detach().cpu().tolist()
                        )
                ]
            )
            data_dict["response"].extend(
                [[{"role": "assistant", "content": seq}] for seq in sequences[i+1]]
            )
            rw = ((rewards[i+1] - rewards[i]).float().detach().cpu() + 5) / 10
            data_dict["reward"].extend(rw.tolist())
            data_dict["system"].extend([""] * len(sequences[i]))
            data_dict["tools"].extend([""] * len(sequences[i]))
        
        return Dataset.from_dict(data_dict)

    def train_policy(
        self,
        dataset,
        data_args,
        callbacks: Optional[List["TrainerCallback"]] = None,
    ) -> None:
        template = get_template_and_fix_tokenizer(self.policy.tokenizer, data_args.template)
        preprocess_func = partial(
            preprocess_unsupervised_dataset, tokenizer=self.policy.tokenizer, template=template, data_args=data_args
        )
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )

        dataset = dataset.map(preprocess_func, batched=True,
                              remove_columns=["prompt", "response", "system", "tools", "reward"], **kwargs)
        
        # Set callbacks
        callbacks = [LogCallback()] if callbacks is None else callbacks

        # use left-padding in generation while using right-padding in training
        self.policy.tokenizer.padding_side = "left"
        self.training_args.remove_unused_columns = False  # important for pairwise dataset

        # Create reference model
        ref_model = create_ref_model(
            self.policy_model_args, self.policy_finetuning_args, add_valuehead=True)

        # Initialize our Trainer
        ppo_trainer = PolicyPPOTrainer(
            model_args=self.policy_model_args,
            training_args=self.training_args,
            finetuning_args=self.policy_finetuning_args,
            generating_args=self.generating_args,
            callbacks=callbacks + [FixValueHeadModelCallback()],
            model=self.policy.model,
            ref_model=ref_model,
            tokenizer=self.policy.tokenizer,
            dataset=dataset,
            data_collator=collate_fn,
        )

        # Training
        if self.training_args.do_train:
            ppo_trainer.ppo_train(
                resume_from_checkpoint=self.training_args.resume_from_checkpoint)
            ppo_trainer.save_model()
            if self.training_args.should_save:
                fix_valuehead_checkpoint(
                    self.policy.model, self.training_args.output_dir, self.training_args.save_safetensors)
            ppo_trainer.save_state()  # must be called after save_model to have a folder
            if ppo_trainer.is_world_process_zero() and self.policy_finetuning_args.plot_loss:
                plot_loss(self.training_args.output_dir,
                          keys=["loss", "reward"])