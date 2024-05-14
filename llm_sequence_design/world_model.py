import os
import gc
import copy
import json
import joblib
import deepspeed
import numpy as np
from tqdm import tqdm
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
from src.llmtuner.train.oracle.metric import compute_regression_metrics
from src.llmtuner.train.oracle.trainer import OracleTrainer

from bayesian_ridge import BayesianRidgeModel
from utils import get_dataset_embedding
from sklearn.linear_model import BayesianRidge

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.trainer import PredictionOutput

    from src.llmtuner.hparams import FinetuningArguments


logger = get_logger(__name__)

class WorldModel:
    def __init__(self, model_args, finetuning_args):
        self.tokenizer = None
        self.model = None
        self.linear_head = None
        self.model_args = model_args
        self.finetuning_args = finetuning_args

    def load(self):
        self.tokenizer = load_tokenizer(self.model_args)
        self.model = load_model(self.tokenizer,
                                self.model_args,
                                self.finetuning_args,
                                is_trainable=False,
                                add_valuehead=False
                                )

    def load_linear_head(self):
        self.linear_head = joblib.load(os.path.join(wm_training_args.output_dir, f"model_{iteration}.joblib"))
        
    def unload(self):
        del self.tokenizer
        del self.model
        self.tokenizer = None
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, X, batch_size = 32, **kwargs):
        total_X = len(X)
        posteriors = []
        
        for i in tqdm(range((total_X//batch_size) + 1)):
            idx_start = i * batch_size
            idx_end = min((i+1) * batch_size, total_X)

            batch_X = X[idx_start:idx_end]
        
            model_inputs = self.tokenizer(batch_X, add_special_tokens=False, return_tensors="pt", padding=True)
            model_inputs = {
                k: v.to(self.model.device) for k, v in model_inputs.items()
            }
            embeds = self.model.model(**model_inputs, **kwargs)

            last_idxs = []
            for i in range(embeds.last_hidden_state.size(0)):
                if self.tokenizer.pad_token_id is None:
                    end_index = -1
                else:
                    end_indexes = (model_inputs["input_ids"][i] != self.tokenizer.pad_token_id).nonzero()
                    end_index = end_indexes[-1].item() if len(end_indexes) else 0
                    
                last_idxs.append(end_index)
                
            embed_last_token = embeds.last_hidden_state[list(range(len(last_idxs))), last_idxs]

            posterior = self.linear_head.posterior(embed_last_token.float())
            posteriors.append(posterior)
            
        return posteriors

    def sample(self, X, sample_size=1, **kwargs):
        posteriors = self.forward(X=X, **kwargs)
        posterior_preds = []
        for batch_posterior in posteriors:
            posterior_pred = batch_posterior.sample(sample_shape=torch.Size([sample_size])).squeeze(-1)
            posterior_preds.append(posterior_pred)
        return torch.concatenate(posterior_preds, dim=1)

    def train_v3(
        self,
        dataset,
        training_args,
        data_args,
        callbacks: Optional[List["TrainerCallback"]] = None,
        iteration = 0,
        **kwargs
    ):
        wm_training_args = copy.deepcopy(training_args)
        wm_training_args.output_dir = os.path.join(
            training_args.output_dir, "world_model")

        dataset_emb = get_dataset_embedding(
            dataset, self.model, self.tokenizer, data_args)

        X_train = dataset_emb.data["inputs_embeds"].to_pylist()
        y_train = dataset_emb.data["rewards"].to_pylist()
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train).reshape(-1, 1)

        self.linear_head = BayesianRidgeModel(X_train, y_train)
        
        os.makedirs(wm_training_args.output_dir, exist_ok=True)
        joblib.dump(self.linear_head, os.path.join(wm_training_args.output_dir, f"model_{iteration}.joblib"))

    def eval_v3(
        self,
        dataset,
        **kwargs
    ):
        X_test = np.stack(eval_dataset["inputs_embeds"])
        y_test = np.stack(eval_dataset["rewards"])

        y_test_hat = self.linear_head.predict(X_test)
        eval_metrics = compute_regression_metrics((y_test_hat, y_test))
        eval_metrics = {f"eval_{k}": v for k, v in eval_metrics.items()}

        return eval_metrics
        
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
        trainer = OracleTrainer(
            model=self.model,
            args=wm_training_args,
            finetuning_args=self.finetuning_args,
            tokenizer=self.tokenizer,
            callbacks=callbacks + [FixValueHeadModelCallback()],
            compute_metrics=compute_regression_metrics,
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
