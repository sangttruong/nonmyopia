from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from llmtuner.model import load_model, load_tokenizer
from llmtuner.hparams import ModelArguments, get_train_args
from llmtuner.data import get_dataset
from tqdm import tqdm
import numpy as np
import joblib
import torch
from typing import Any, TYPE_CHECKING, Dict, List, Optional
import os
import gc
import json
from utils import compute_regression_metrics
from llmtuner.extras.callbacks import LogCallback
from datasets import load_dataset

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback


class Oracle:
    def __init__(self, model_args, finetuning_args):
        self.tokenizer = None
        self.model = None
        self.model_args = model_args
        self.finetuning_args = finetuning_args

        assert model_args.linear_head_path, "--oracle_linear_head_path must be defined!"
        self.linear_head = joblib.load(os.path.join(
            model_args.oracle_linear_head_path, 'model.joblib'))

    def load(self):
        self.tokenizer = load_tokenizer(self.model_args)
        self.model = load_model(self.tokenizer,
                                self.model_args,
                                self.finetuning_args,
                                is_trainable=False,
                                add_valuehead=False
                                )

    def unload(self):
        del self.tokenizer
        del self.model
        self.tokenizer = None
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, X, batch_size=32, **kwargs):
        total_X = len(X)
        outputs = []

        total_steps = total_X // batch_size
        if total_steps * batch_size < total_X:
            total_steps += 1

        for i in tqdm(range(total_steps)):
            idx_start = i * batch_size
            idx_end = min((i+1) * batch_size, total_X)

            batch_X = X[idx_start:idx_end]

            model_inputs = self.tokenizer(
                batch_X, add_special_tokens=False, return_tensors="pt", padding=True)
            model_inputs = {
                k: v.to(self.model.device) for k, v in model_inputs.items()
            }
            embeds = self.model.model(**model_inputs, **kwargs)

            last_idxs = []
            for i in range(embeds.last_hidden_state.size(0)):
                if self.tokenizer.pad_token_id is None:
                    end_index = -1
                else:
                    end_indexes = (
                        model_inputs["input_ids"][i] != self.tokenizer.pad_token_id).nonzero()
                    end_index = end_indexes[-1].item() if len(end_indexes) else 0

                last_idxs.append(end_index)

            embed_last_token = embeds.last_hidden_state[list(
                range(len(last_idxs))), last_idxs]

            y_mean = self.linear_head.predict(
                embed_last_token.float().cpu().detach().numpy())
            outputs.extend(y_mean.tolist())

        return outputs


def run_oracle(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    if not os.path.exists(model_args.model_name_or_path):
        if model_args.model_name_or_path.lower() == "linear":
            model = LinearRegression()
        elif model_args.model_name_or_path.lower() == "ridge":
            model = Ridge(alpha=10.0)
        elif model_args.model_name_or_path.lower() == "bayesridge":
            model = BayesianRidge()
        else:
            raise ValueError(
                f"model_name_or_path {model_args.model_name_or_path} is not supported"
            )
    else:
        model = joblib.load(os.path.join(
            model_args.model_name_or_path, 'model.joblib'))

    # Training
    if training_args.do_train:
        print("Training oracle model...")
        data_args.split = "train"
        train_dataset = load_dataset(data_args.dataset)[data_args.split]
        # train_dataset = load_dataset(
        #     None, model_args, data_args, training_args, stage="oracle",)

        X_train = np.stack(train_dataset["inputs_embeds"])
        y_train = np.stack(train_dataset["rewards"])
        model.fit(X_train, y_train)

        # Save model
        os.makedirs(training_args.output_dir, exist_ok=True)
        joblib.dump(model, os.path.join(
            training_args.output_dir, 'model.joblib'))

        # Save results
        y_train_hat = model.predict(X_train)
        train_metrics = compute_regression_metrics((y_train_hat, y_train))
        train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        with open(os.path.join(training_args.output_dir, 'train_results.json'), 'w') as f:
            json.dump(train_metrics, f)

    # Evaluation
    if training_args.do_eval:
        print("Evaluating oracle model...")
        data_args.split = "validation"
        eval_dataset = load_dataset(data_args.dataset)[data_args.split]

        X_test = np.stack(eval_dataset["inputs_embeds"])
        y_test = np.stack(eval_dataset["rewards"])

        # Save results
        y_test_hat = model.predict(X_test)
        eval_metrics = compute_regression_metrics((y_test_hat, y_test))
        eval_metrics = {f"eval_{k}": v for k, v in eval_metrics.items()}
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, 'eval_results.json'), 'w') as f:
            json.dump(eval_metrics, f)

    # Predict
    if training_args.do_predict:
        print("Predicting oracle model...")
        data_args.split = "validation"
        eval_dataset = load_dataset(data_args.dataset)[data_args.split]

        X_test = np.array(eval_dataset.data["inputs_embeds"])
        y_test = np.array(eval_dataset.data["rewards"])

        # Save to jsonl file
        y_test_hat = model.predict(X_test)
        with open(os.path.join(training_args.output_dir, 'predictions.jsonl'), 'w') as f:
            for i in range(len(y_test)):
                json.dump({"inputs_embeds": X_test[i].tolist(
                ), "rewards": y_test[i], "rewards_hat": y_test_hat[i]}, f)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks
    run_oracle(model_args, data_args, training_args, finetuning_args, callbacks)

if __name__ == "__main__":
    run_exp()
