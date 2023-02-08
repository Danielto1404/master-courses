import typing as tp

import numpy as np

from .tensor import Tensor, tensor


def randn(*dn: int, norm=1.0) -> Tensor:
    array = np.random.randn(*dn) / norm
    return tensor(array)


def uniform(
        size: tp.Tuple[int, ...],
        low: tp.Union[float, int] = 0.0,
        high: tp.Union[float, int] = 1.0
) -> Tensor:
    array = np.random.uniform(low, high, size)
    return tensor(array)


def ones(*dn: int) -> Tensor:
    array = np.ones(dn)
    return tensor(array)


def zeros(*dn: int) -> Tensor:
    array = np.zeros(dn)
    return tensor(array)


def zeros_like(t: Tensor | np.ndarray):
    return zeros(*t.shape)


def ones_like(t: Tensor | np.ndarray):
    return ones(*t.shape)


def grad_clip(parameters, max_value: float):
    assert max_value > 0, "max_value must be a positive number"
    for p in parameters:
        if p.grad is not None:
            p.grad = np.clip(p.grad, -max_value, max_value)


def to_pair(a):
    return a if isinstance(a, tuple) else (a, a)


tab = "\t"  # tab symbol
nl = "\n"  # new line symbol

__all__ = [
    "tensor",
    "randn",
    "uniform",
    "ones",
    "zeros",
    "ones_like",
    "zeros_like",
    "grad_clip",
    "to_pair"
]
