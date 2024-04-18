from typing import Any

import torch
from src.llmtuner.hparams import ModelArguments
from src.llmtuner.model import load_model, load_tokenizer


class Oracle:
    def __init__(self, model_args, finetuning_args):
        model_args = ModelArguments(
            model_name_or_path=model_args.oracle_model_name_or_path,
            adapter_name_or_path=model_args.oracle_adapter_name_or_path,
            use_fast_tokenizer=model_args.oracle_use_fast_tokenizer,
            flash_attn=model_args.oracle_flash_attn,
            quantization_bit=model_args.oracle_quantization_bit,
            quantization_type=model_args.oracle_quantization_type,
            quantization_device_map=model_args.oracle_quantization_device_map,
            use_unsloth=model_args.oracle_use_unsloth
        )
        self.tokenizer = load_tokenizer(model_args)
        self.__model__ = load_model(self.tokenizer,
                                    model_args,
                                    finetuning_args,
                                    is_trainable=False,
                                    add_valuehead=True
                                    )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, X, *args, **kwargs):
        X_tokenized = self.tokenizer.encode(X, add_special_tokens=False)
        model_inputs = {
            "input_ids": X_tokenized,
            "attention_mask": torch.one_likes(X_tokenized),
        }

        return self.__model__(model_inputs, *args, **kwargs)
