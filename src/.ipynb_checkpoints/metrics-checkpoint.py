from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
   
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    err = y_true - y_pred
    return float(np.mean(err * err))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
 
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    m = mse(y_true, y_pred)
    var = float(np.var(y_true))
    return float(np.sqrt(m / (var + eps))) #    eps avoids division by zero if var is extremely small.
