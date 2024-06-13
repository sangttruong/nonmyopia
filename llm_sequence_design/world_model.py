import os
import gc
import copy
import joblib
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Any

import torch

from src.llmtuner.model import load_model, load_tokenizer
from src.llmtuner.extras.logging import get_logger

from bayesian_ridge import BayesianRidgeModel
from utils import get_dataset_embedding, compute_regression_metrics

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
        posteriors = []

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

            posterior = self.linear_head.posterior(embed_last_token.float())
            posteriors.append(posterior)

        return posteriors

    def sample(self, X, sample_size=1, **kwargs):
        posteriors = self.forward(X=X, **kwargs)
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
        joblib.dump(self.linear_head, os.path.join(
            wm_training_args.output_dir, f"model_{iteration}.joblib"))

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
