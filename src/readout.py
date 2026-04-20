# src/readout.py
"""
Readout / decoding methods 

Currently:
- Ridge regression readout (linear readout)

Later:
- Lasso / ElasticNet
- nonlinear readouts (MLP, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class SplitData:
    R_train: np.ndarray
    R_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def split_states_targets(R: np.ndarray, y: np.ndarray, split: float = 0.8,) -> SplitData:
 
    R = np.asarray(R)
    y = np.asarray(y)

    if R.ndim != 2:
        raise ValueError("R must be 2D with shape (L, N).")
    if y.ndim != 1:
        raise ValueError("y must be 1D with shape (L,).")
    if R.shape[0] != y.shape[0]:
        raise ValueError(f"R and y must have the same length L. Got {R.shape[0]} vs {y.shape[0]}.")
        

    idx = int(split * R.shape[0])
    return SplitData(R_train=R[:idx], R_test=R[idx:], y_train=y[:idx], y_test=y[idx:],)

#===

#def perform_washout(R:np.ndarray, y: np.ndarray, n_of_cycles: int): 

    




#======================================================RIDGE==============================================================================================

def fit_ridge_readout(R_train: np.ndarray, y_train: np.ndarray, alpha: float = 1e-4, *, fit_intercept: bool = False,) -> Ridge: #The ridge class is defined above when imported  # FUNCTION MAY or MAY NOT BE NECCESARY as this function is already defined in class ridge....

    model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(R_train, y_train)
    return model


def predict_readout(model: Ridge, R: np.ndarray) -> np.ndarray: # FUNCTION MAY or MAY NOT BE NECCESARY as this 
   
    return model.predict(R)


def fit_and_predict_ridge(R: np.ndarray, y: np.ndarray, split: float = 0.8, alpha: float = 1e-4, *, fit_intercept: bool = False,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Ridge]:

    s = split_states_targets(R, y, split=split)
    model = fit_ridge_readout(s.R_train, s.y_train, alpha=alpha, fit_intercept=fit_intercept)

    y_pred_train = predict_readout(model, s.R_train)
    y_pred_test = predict_readout(model, s.R_test)

    return y_pred_train, y_pred_test, s.y_train, s.y_test, model

#==========================================================================================================================================================


