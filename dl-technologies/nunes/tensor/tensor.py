import typing as tp

import numpy as np
import torch

import nunes.tensor.math as numath
from nunes.autograd.graph import Graph


def tensor(
        data: tp.Union[tp.List, np.ndarray, int, float, np.float32, np.float64, np.int32, np.int64],
        requires_grad: bool = False
) -> "Tensor":
    if isinstance(data, (int, float, np.float32, np.float64, np.int32, np.int64)):
        data = np.array(data)
    elif isinstance(data, list):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise NotImplementedError(
            "Not supported data type. Please provide one of this: [list, numpy.array, int, float]"
        )

    return Tensor(data, requires_grad=requires_grad)


def from_torch(t: torch.Tensor, requires_grad: bool = False) -> "Tensor":
    return tensor(t.numpy(), requires_grad=requires_grad)


class Tensor:
    """Numpy multidimensional array wrapper which supports gradient calculation"""

    def __init__(
            self,
            data: np.ndarray,
            requires_grad: bool = False
    ):
        self.data = data
        self.grad: tp.Optional[np.ndarray] = None
        self.graph = Graph() if requires_grad else None

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        return self.data.shape

    def numpy(self) -> np.ndarray:
        return self.data

    def is_single(self) -> bool:
        return len(self.shape) == 0

    def item(self):
        return self.data.item()

    def as_parameter(self, requires_grad: bool = True) -> "Parameter":
        return Parameter(data=self.data, requires_grad=requires_grad)

    @property
    def requires_grad(self) -> bool:
        return self.graph is not None

    def backward(self):
        assert self.requires_grad, "Can't call `backward` method because this tensor doesn't requires gradient"
        self.graph.backward()

    def numel(self) -> int:
        return self.data.size

    def sum(self) -> "Tensor":
        return numath.sum(self)

    def log(self) -> "Tensor":
        return numath.log(self)

    def mean(self) -> "Tensor":
        return numath.mean(self)

    def exp(self) -> "Tensor":
        return numath.exp(self)

    def reshape(self, shape: tp.Tuple[int, ...]) -> "Tensor":
        return numath.reshape(self, shape)

    def __add__(self, other) -> "Tensor":
        if isinstance(other, (float, int)):
            other = tensor(other)

        return numath.add(self, other)

    def __mul__(self, other) -> "Tensor":
        if isinstance(other, (float, int)):
            other = tensor(other)

        return numath.mul(self, other)

    def __truediv__(self, other) -> "Tensor":
        if isinstance(other, (float, int)):
            other = tensor(other)
        return numath.div(self, other)

    def __sub__(self, other) -> "Tensor":
        if isinstance(other, (float, int)):
            other = tensor(other)
        return numath.sub(self, other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return numath.matmul(self, other)

    def __neg__(self) -> "Tensor":
        return numath.negate(self)

    def __pow__(self, power, modulo=None) -> "Tensor":
        return numath.pow(self, times=power)

    def __getitem__(self, *slices) -> "Tensor":
        return numath.indices_slice(self, *slices)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        data = repr(self.data).replace("array", "").replace("(", "").replace(")", "")
        return f"tensor({data}, requires_grad={self.requires_grad})"


class Parameter(Tensor):
    """`Tensor` wrapper for using in `nn.Module` blocks."""
    pass


__all__ = [
    "Tensor",
    "Parameter",
    "tensor",
    "from_torch"
]
