from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch.nn.functional as F


def compute_accuracy(eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
    preds, _ = eval_preds
    return {"accuracy": (preds[0] > preds[1]).sum() / len(preds[0])}


def compute_rmse(eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
    preds, _ = eval_preds
    return {"rmse": F.mse_loss(preds[0], preds[1], reduction="mean").sqrt().item()}
