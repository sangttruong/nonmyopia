from typing import Any
import os
import gc
import torch
import joblib
from tqdm import tqdm
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
        total_X = len(X)
        batch_size = 32
        outputs = []
        
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

            y_mean = self.linear_head.predict(embed_last_token.float().cpu().detach().numpy())
            outputs.extend(y_mean.tolist())
            
        return outputs
