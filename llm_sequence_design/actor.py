import os
import copy
from transformers import DataCollatorWithPadding
from typing import TYPE_CHECKING, List, Optional

from src.llmtuner.extras.callbacks import FixValueHeadModelCallback
from src.llmtuner.extras.misc import fix_valuehead_checkpoint
from src.llmtuner.extras.ploting import plot_loss
from src.llmtuner.train.utils import create_ref_model
from acqfs import (
    acqf_hes,
)
from policy import Policy, PolicyPPOTrainer


class Actor:
    def __init__(self, bo_args, policy_model_args, policy_finetuning_args, training_args, generating_args):
        self.policy = Policy(policy_model_args, policy_finetuning_args)

        self.bo_args = bo_args
        self.policy_model_args = policy_model_args
        self.policy_finetuning_args = policy_finetuning_args
        self.training_args = copy.deepcopy(training_args)
        self.training_args.output_dir = os.path.join(
            self.training_args.output_dir, "policy")

        self.generating_args = generating_args

        self.algo_lookahead_steps = bo_args.algo_lookahead_steps

        if self.bo_args.algo == "HES":
            self.acqf = acqf_hes
        else:
            raise NotImplementedError

    def query(self, dataset, reward_model):
        self.train_policy(dataset, reward_model)

        # Query the next sequence
        breakpoint()
        previous_x = dataset.data[-1]

    def train_policy(
        self,
        dataset,
        reward_model,
        callbacks: Optional[List["TrainerCallback"]] = None,
    ) -> None:
        # use left-padding in generation while using right-padding in training
        self.policy.tokenizer.padding_side = "left"
        data_collator = DataCollatorWithPadding(
            tokenizer=self.policy.tokenizer)

        # Create reference model and reward model
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
            reward_model=reward_model,
            ref_model=ref_model,
            tokenizer=self.policy.tokenizer,
            dataset=dataset,
            data_collator=data_collator,
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
