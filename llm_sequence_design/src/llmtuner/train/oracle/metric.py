from typing import Dict, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

def compute_regression_metrics(eval_preds: Sequence[Union[np.array, Tuple[np.array]]]) -> Dict[str, float]:
    preds, labels = eval_preds
    labels = labels.reshape(-1, 1)
    
    return {"mae": mean_absolute_error(labels, preds),
            "r2": r2_score(labels, preds), 
            "rmse": root_mean_squared_error(labels, preds)}
