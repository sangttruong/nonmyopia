from typing import Any

import torch
from src.llmtuner.hparams import ModelArguments
from src.llmtuner.model import load_model, load_tokenizer


class Oracle:
    def __init__(self, model_args, finetuning_args):
        self.tokenizer = load_tokenizer(model_args)
        self.model = load_model(self.tokenizer,
                                model_args,
                                finetuning_args,
                                is_trainable=False,
                                add_valuehead=False
                                )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, X, *args, **kwargs):
        X_tokenized = self.tokenizer.encode(X, add_special_tokens=False)
        model_inputs = {
            "input_ids": X_tokenized,
            "attention_mask": torch.one_likes(X_tokenized),
        }

        return self.model(model_inputs, *args, **kwargs)
