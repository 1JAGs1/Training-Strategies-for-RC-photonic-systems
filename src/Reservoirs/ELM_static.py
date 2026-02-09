from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass(frozen=True)
class ELMStaticParams:

    g: float = 1.0  # input gain / scaling before tanh


def run_elm_static(V: np.ndarray, params: Optional[ELMStaticParams] = None, *, mode: Literal["vectorised", "loop"] = "vectorised",
) -> np.ndarray:

    V = np.asarray(V, dtype=float)
    if V.ndim != 2:
        raise ValueError("V must be 2D with shape (L, N).")

    if params is None:
        params = ELMStaticParams()

    g = float(params.g)

    if mode == "vectorised":
        return np.tanh(g * V)

    if mode == "loop":
        L, N = V.shape
        R = np.zeros((L, N), dtype=float)
        for l in range(L):
            for n in range(N):
                R[l, n] = np.tanh(g * V[l, n])
        return R

    raise ValueError(f"Unknown mode: {mode}")
