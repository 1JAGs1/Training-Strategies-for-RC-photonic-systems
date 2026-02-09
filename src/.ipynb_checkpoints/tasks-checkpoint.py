from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TaskData:
   
    x: np.ndarray
    y: np.ndarray


def make_static_sin_task(L: int = 2000, seed: int = 0, low: float = -1.0, high: float = 1.0,) -> TaskData:  # saying that the function should return type "TaskData"(defined above)

    rng = np.random.default_rng(seed)
    x = rng.uniform(low, high, size=L)
    y = np.sin(2.0 * np.pi * x)
    return TaskData(x=x, y=y)


def train_test_split_1d(x: np.ndarray, y: np.ndarray, split: float = 0.8,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("train_test_split_1d expects 1D arrays x and y.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    idx = int(split * len(x))
    return x[:idx], x[idx:], y[:idx], y[idx:]
