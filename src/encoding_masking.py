from __future__ import annotations

from dataclasses import dataclass  # python is a fake language. this saves you from doing "init" part of creating a class.

from typing import Optional, Union #when defining types, "optional" and "union" help with flexibility when using types.

import numpy as np


ArrayLike = Union[np.ndarray]

'''
def generate_mask(N: int, rng: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
 
    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)

    return gen.random(N)
'''
#the below is from -1 to 1 as opposed to 0 to 1, we might have to do prior tests to see if bettter resposne 

def generate_mask(N: int, rng: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
 
    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)

    return gen.random(N)




def standardise(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
   
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sigma = np.std(x)
    return (x - mu) / (sigma + eps) # eps avoids division by 0 error


def apply_mask(x_norm: np.ndarray, mask: np.ndarray) -> np.ndarray:
 
    x_norm = np.asarray(x_norm, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=float).reshape(-1)
    return np.outer(x_norm, mask)


@dataclass(frozen=True)
class TDMParams:
   
    N: int
    theta: float
    dt: float
    T: float
    tau: float
    Nd_loop: int
    Nd_delay:int
    tap_stride: int


def compute_tdm_params(
    N: int,
    theta: float,
    dt: float,
    tau_factor: float = 1.01,
) -> TDMParams:
    
    T = N * theta
    tau = tau_factor * T
    Nd_loop = int(round(T / dt))
    Nd_delay = int(round(tau / dt))
    tap_stride = int(round(theta / dt))

    if Nd_delay <= 0:
        raise ValueError(f"Nd computed as {Nd_delay}. Check tau/dt (tau={tau}, dt={dt}).")
    if tap_stride <= 0:
        raise ValueError(
            f"tap_stride computed as {tap_stride}. Check theta/dt (theta={theta}, dt={dt})."
        )

    return TDMParams(N=N, theta=theta, dt=dt, T=T, tau=tau, Nd_loop=Nd_loop, Nd_delay=Nd_delay, tap_stride=tap_stride)


def build_masked_input(x: np.ndarray, N: int, rng: Optional[Union[int, np.random.Generator]] = None, *, do_standardise: bool = True,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
 
    mask = generate_mask(N, rng=rng)
    x_norm = standardise(x) if do_standardise else np.asarray(x, dtype=float)
    V = apply_mask(x_norm, mask)
    return x_norm, mask, V
