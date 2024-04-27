import os
import copy
import json
from functools import partial
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, Any

import torch
from transformers import Trainer

from src.llmtuner.hparams import ModelArguments
from src.llmtuner.model import load_model, load_tokenizer
from src.llmtuner.extras.logging import get_logger
from src.llmtuner.train.utils import create_custom_optimzer, create_custom_scheduler
from src.llmtuner.data.preprocess import preprocess_oracle_dataset
from src.llmtuner.extras.callbacks import FixValueHeadModelCallback
from src.llmtuner.extras.misc import fix_valuehead_checkpoint
from src.llmtuner.extras.ploting import plot_loss
from src.llmtuner.train.oracle.metric import compute_rmse

from utils import get_dataset_embedding

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.trainer import PredictionOutput

    from src.llmtuner.hparams import FinetuningArguments


logger = get_logger(__name__)


class WorldModel:
    def __init__(self, model_args, finetuning_args):
        self.tokenizer = load_tokenizer(model_args)
        self.model = load_model(self.tokenizer,
                                model_args,
                                finetuning_args,
                                is_trainable=True,
                                add_valuehead=True
                                )
        self.finetuning_args = finetuning_args

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, X, *args, **kwargs):
        X_tokenized = self.tokenizer.encode(X, add_special_tokens=False)
        model_inputs = {
            "input_ids": X_tokenized,
            "attention_mask": torch.one_likes(X_tokenized),
        }

        return self.model(model_inputs, *args, **kwargs)

    def train_v2(
        self,
        dataset,
        training_args,
        data_args,
        callbacks: Optional[List["TrainerCallback"]] = None,
        **kwargs
    ):
        wm_training_args = copy.deepcopy(training_args)
        wm_training_args.output_dir = os.path.join(
            training_args.output_dir, "world_model")

        dataset_emb = get_dataset_embedding(
            dataset, self.model, self.tokenizer, data_args)

        backward_batch_size = training_args.per_device_train_batch_size * \
            training_args.gradient_accumulation_steps
        dataset_loader = torch.utils.data.DataLoader(
            dataset_emb, batch_size=backward_batch_size, shuffle=True)

        linear_model = torch.nn.Linear(
            in_features=self.model.pretrained_model.model.config.hidden_size, out_features=1)

        optimizer = torch.optim.AdamW(
            linear_model.parameters(), lr=wm_training_args.learning_rate)

        for epoch in range(int(wm_training_args.num_train_epochs)):
            for batch in dataset_loader:
                X = torch.stack(batch['inputs_embeds']).float().T
                y = batch['rewards'].reshape(-1, 1).float()
                y_pred = linear_model(X)
                loss = torch.nn.functional.mse_loss(y_pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

        self.model.v_head.summary.load_state_dict(linear_model.state_dict())
        self.model.save_pretrained(
            wm_training_args.output_dir, safe_serialization=wm_training_args.save_safetensors)
        if training_args.should_save:
            fix_valuehead_checkpoint(
                self.model, wm_training_args.output_dir, wm_training_args.save_safetensors)

    def train(
        self,
        dataset,
        training_args,
        data_args,
        callbacks: Optional[List["TrainerCallback"]] = None,
        **kwargs
    ):
        # Dataset has loaded to correct column names
        # Now we can preprocess the dataset
        with training_args.main_process_first(desc="pre-process dataset"):
            preprocess_func = partial(
                preprocess_oracle_dataset, tokenizer=self.tokenizer, template=data_args.template, data_args=data_args
            )
            column_names = list(next(iter(dataset)).keys())
            kwargs = {}
            if not data_args.streaming:
                kwargs = dict(
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=(not data_args.overwrite_cache),
                    desc="Running tokenizer on dataset",
                )

            dataset = dataset.map(preprocess_func, batched=True,
                                  remove_columns=column_names, **kwargs)
            if data_args.tokenized_path is not None:
                if training_args.should_save:
                    dataset.save_to_disk(data_args.tokenized_path)
                    logger.info("Tokenized dataset saved at {}.".format(
                        data_args.tokenized_path))
                    logger.info(
                        "Please restart the training with `--tokenized_path {}`.".format(data_args.tokenized_path))

                exit(0)

        # Update arguments
        wm_training_args = copy.deepcopy(training_args)
        wm_training_args.output_dir = os.path.join(
            training_args.output_dir, "world_model")
        wm_training_args.remove_unused_columns = False  # important for pairwise dataset

        # Initialize our Trainer
        trainer = WMTrainer(
            model=self.model,
            args=wm_training_args,
            finetuning_args=self.finetuning_args,
            tokenizer=self.tokenizer,
            callbacks=callbacks + [FixValueHeadModelCallback()],
            compute_metrics=compute_rmse,
            train_dataset=dataset,
        )

        # Training
        train_result = trainer.train(
            resume_from_checkpoint=wm_training_args.resume_from_checkpoint)
        trainer.save_model()
        if wm_training_args.should_save:
            fix_valuehead_checkpoint(
                self.model, wm_training_args.output_dir, wm_training_args.save_safetensors)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and self.finetuning_args.plot_loss:
            plot_loss(wm_training_args.output_dir, keys=[
                "loss", "eval_loss", "eval_rmse"])


class WMTrainer(Trainer):
    r"""
    Inherits Trainer to compute pairwise loss.
    """

    def __init__(self, finetuning_args: "FinetuningArguments", **kwargs) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.can_return_loss = True  # override property to return eval_loss

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(
                self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def compute_loss(
        self,
        model: "PreTrainedModel",
        inputs: Dict[str, torch.Tensor],
        return_outputs: Optional[bool] = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        r"""
        Computes loss.

        Subclass and override to inject custom behavior.

        Note that the first element will be removed from the output tuple.
        See: https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/trainer.py#L3509
        """
        # Compute rewards
        _, _, values = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True)

        unwrapped_model: "PreTrainedModel" = self.accelerator.unwrap_model(
            self.model)
        if getattr(unwrapped_model.config, "model_type", None) == "chatglm":
            values = torch.transpose(values, 0, 1)

        # Split the inputs and rewards into two parts, chosen and rejected
        batch_size = inputs["input_ids"].size(0)
        input_ids = inputs["input_ids"]
        real_rewards = inputs["rewards"]
        rewards = values
        scores = []

        # Compute pairwise loss. Only backprop on the different tokens before padding
        # Inspired by: https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
        loss = 0
        for i in range(batch_size):
            end_index = length = (
                input_ids[i] != self.tokenizer.pad_token_id).nonzero()[-1] + 1
            div_index = end_index - 1

            assert div_index > 0
            trunc_rewards = rewards[i, div_index:end_index]
            if return_outputs:  # use the score on the last token except pad token for inference
                scores.append(rewards[i, length-1])
            loss += torch.nn.functional.mse_loss(trunc_rewards,
                                                 real_rewards[i].unsqueeze(0),
                                                 reduction='mean').mean()

        loss = loss / batch_size
        if return_outputs:
            scores = torch.stack(scores)
            return loss, [loss, scores]

        return loss

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(
            self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")
        chosen_scores, rejected_scores = predict_results.predictions

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for c_score, r_score in zip(chosen_scores, rejected_scores):
                res.append(json.dumps(
                    {"chosen": round(float(c_score), 2), "rejected": round(float(r_score), 2)}))
            writer.write("\n".join(res))
