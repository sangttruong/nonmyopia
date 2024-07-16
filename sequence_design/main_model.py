import os
import gc
import copy
import json
import torch
import joblib
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from typing import Any, TYPE_CHECKING, Dict, List, Optional

from llmtuner.model import load_model, load_tokenizer
from llmtuner.hparams import ModelArguments, get_train_args
from llmtuner.extras.logging import get_logger
from llmtuner.extras.callbacks import LogCallback

from bayesian_ridge import BayesianRidgeModel
from utils import get_dataset_embedding, compute_regression_metrics
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

logger = get_logger(__name__)


class MainModel:
    def __init__(self, model_args, finetuning_args):
        self.tokenizer = None
        self.model = None
        self.linear_head = None
        self.model_args = model_args
        self.finetuning_args = finetuning_args

        if model_args.linear_head_path:
            print("Loading linear head of model...")
            self.linear_head = joblib.load(model_args.linear_head_path)

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
        # Changed to self.predict, instead of self.forward, which is now
        # self.posterior_predictive
        return self.predict(*args, **kwargs)

    def posterior_predictive(self, X, batch_size=32, **kwargs):
        # former surr_model forward
        total_X = len(X)
        posteriors = []

        total_steps = total_X // batch_size
        if total_steps * batch_size < total_X:
            total_steps += 1

        for i in tqdm(range(total_steps)):
            idx_start = i * batch_size
            idx_end = min((i+1) * batch_size, total_X)

            batch_X = X[idx_start:idx_end]

            model_inputs = self.tokenizer(
                batch_X, add_special_tokens=False,
                return_tensors="pt", padding=True)
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
                        model_inputs["input_ids"][i] != self.
                        tokenizer.pad_token_id).nonzero()
                    end_index = end_indexes[-1].item()\
                        if len(end_indexes) else 0

                last_idxs.append(end_index)

            embed_last_token = embeds.last_hidden_state[list(
                range(len(last_idxs))), last_idxs]

            posterior = self.linear_head.posterior(embed_last_token.float())
            posteriors.append(posterior)

        return posteriors

    def predict(self, X, batch_size=32, **kwargs):
        # former Oracle forward
        total_X = len(X)
        outputs = []

        total_steps = total_X // batch_size
        if total_steps * batch_size < total_X:
            total_steps += 1

        for i in tqdm(range(total_steps)):
            idx_start = i * batch_size
            idx_end = min((i + 1) * batch_size, total_X)

            batch_X = X[idx_start:idx_end]

            model_inputs = self.tokenizer(
                batch_X,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True
            )
            model_inputs = {k: v.to(self.model.device)
                            for k, v in model_inputs.items()}
            embeds = self.model.model(**model_inputs, **kwargs)

            last_idxs = []
            for i in range(embeds.last_hidden_state.size(0)):
                if self.tokenizer.pad_token_id is None:
                    end_index = -1
                else:
                    end_indexes = (
                        model_inputs["input_ids"][i] != self.
                        tokenizer.pad_token_id
                    ).nonzero()
                    end_index = end_indexes[-1].item() \
                        if len(end_indexes) else 0

                last_idxs.append(end_index)

            embed_last_token = embeds.last_hidden_state[
                list(range(len(last_idxs))), last_idxs
            ]

            y_mean = self.linear_head.predict(
                embed_last_token.float().cpu().detach().numpy()
            )
            outputs.extend(y_mean.tolist())

        return outputs

    def sample(self, X, sample_size=1, **kwargs):
        posteriors = self.o(X=X, **kwargs)
        posterior_preds = []
        for batch_posterior in posteriors:
            posterior_pred = batch_posterior.sample(
                sample_shape=torch.Size([sample_size])).squeeze(-1)
            posterior_preds.append(posterior_pred)
        return torch.concatenate(posterior_preds, dim=1)

    def train(
        self,
        dataset,
        training_args,
        data_args,
        callbacks: Optional[List["TrainerCallback"]] = None,
        iteration=0,
        **kwargs
    ):
        surr_training_args = copy.deepcopy(training_args)
        surr_training_args.output_dir = os.path.join(
            training_args.output_dir, "surr_model")

        dataset_emb = get_dataset_embedding(
            dataset, self.model, self.tokenizer, data_args)

        X_train = dataset_emb.data["inputs_embeds"].to_pylist()
        y_train = dataset_emb.data["rewards"].to_pylist()
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train).reshape(-1, 1)

        self.linear_head = BayesianRidgeModel(X_train, y_train)

        os.makedirs(surr_training_args.output_dir, exist_ok=True)
        joblib.dump(self.linear_head, os.path.join(
            surr_training_args.output_dir, f"model_{iteration}.joblib"))

    def eval(
        self,
        dataset,
        **kwargs
    ):
        X_test = np.stack(dataset["inputs_embeds"])
        y_test = np.stack(dataset["rewards"])

        y_test_hat = self.linear_head.predict(X_test)
        eval_metrics = compute_regression_metrics((y_test_hat, y_test))
        eval_metrics = {f"eval_{k}": v for k, v in eval_metrics.items()}

        return eval_metrics


def run_model(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    model = None
    if model_args.model_name_or_path.lower() != "bayesridge":
        raise ValueError(
            f"model_name_or_path {model_args.model_name_or_path} \
            is not supported"
        )

    # Training
    if training_args.do_train:
        print("Training model...")
        data_args.split = "train"
        train_dataset = load_dataset(data_args.dataset)[data_args.split]

        X_train = torch.tensor(train_dataset["inputs_embeds"])
        y_train = torch.tensor(train_dataset["rewards"]).reshape(-1, 1)
        model = BayesianRidgeModel(X_train, y_train)

        # Save model
        os.makedirs(training_args.output_dir, exist_ok=True)
        joblib.dump(model, os.path.join(
            training_args.output_dir, "model.joblib"))

        # Save results
        y_train_hat = model.predict(X_train)

        train_metrics = compute_regression_metrics(
            (y_train_hat, y_train.numpy()))
        train_metrics = {f"train_{k}": float(v)
                         for k, v in train_metrics.items()}
        with open(
            os.path.join(training_args.output_dir, "train_results.json"), "w"
        ) as f:
            json.dump(train_metrics, f)

    # Evaluation
    if training_args.do_eval:
        print("Evaluating oracle model...")
        data_args.split = "validation"
        eval_dataset = load_dataset(data_args.dataset)[data_args.split]

        X_test = torch.tensor(eval_dataset["inputs_embeds"])
        y_test = torch.tensor(eval_dataset["rewards"]).reshape(-1, 1)

        # Save results
        y_test_hat = model.predict(X_test)
        y_test_dist = model.posterior(X_test)
        y_test_hat = y_test_dist.sample(
            sample_shape=torch.Size([1])).mean(dim=0).numpy()

        eval_metrics = compute_regression_metrics((y_test_hat, y_test.numpy()))
        eval_metrics = {f"eval_{k}": float(v) for k, v in eval_metrics.items()}
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(
            os.path.join(training_args.output_dir, "eval_results.json"), "w"
        ) as f:
            json.dump(eval_metrics, f)

    # Predict
    if training_args.do_predict:
        print("Predicting oracle model...")
        data_args.split = "validation"
        eval_dataset = load_dataset(data_args.dataset)[data_args.split]

        X_test = torch.tensor(eval_dataset["inputs_embeds"])
        y_test = torch.tensor(eval_dataset["rewards"]).reshape(-1, 1)

        # Save to jsonl file
        y_test_hat = model.predict(X_test)

        with open(
            os.path.join(training_args.output_dir, "predictions.jsonl"), "w"
        ) as f:
            for i in range(len(y_test)):
                json.dump(
                    {
                        "inputs_embeds": X_test[i].tolist(),
                        "rewards": y_test[i],
                        "rewards_hat": y_test_hat[:, i],
                    },
                    f,
                )


def run_exp(
    args: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    model_args, data_args, training_args, finetuning_args, _ = (
        get_train_args(args)
    )
    callbacks = [LogCallback()] if callbacks is None else callbacks
    run_model(model_args, data_args, training_args,
              finetuning_args, callbacks)


if __name__ == "__main__":
    run_exp()
