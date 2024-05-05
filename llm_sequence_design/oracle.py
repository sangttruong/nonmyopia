from typing import Any
import os
import gc
import torch
import joblib
from src.llmtuner.hparams import ModelArguments
from src.llmtuner.model import load_model, load_tokenizer
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge

class Oracle:
    def __init__(self, model_args, finetuning_args):
        self.tokenizer = None
        self.model = None
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        
        assert model_args.oracle_linear_head_path, "--oracle_linear_head_path must be defined!" 
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

    def forward(self, X, *args, **kwargs):
        X_tokenized = self.tokenizer.encode(X, add_special_tokens=False)
        model_inputs = {
            "input_ids": X_tokenized,
            "attention_mask": torch.one_likes(X_tokenized),
        }
        output = self.model(model_inputs, *args, **kwargs)
        output = self.linear_head.predict(output.last_hidden_state)
        return output
