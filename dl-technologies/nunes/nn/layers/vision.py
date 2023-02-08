import typing as tp

import nunes

from ...tensor import utils as nutils
from ..module import Module

_int2d = tp.Union[int, tp.Tuple[int, int]]


class MaxPool2D(Module):
    def __init__(
            self,
            kernel_size: tp.Union[int, tp.Tuple[int, int]] = (2, 2),
            stride: tp.Union[int, tp.Tuple[int, int]] = (2, 2)
    ):
        super().__init__()
        self.kernel_size = nutils.to_pair(kernel_size)
        self.stride = nutils.to_pair(stride)

    def forward(self, x: nunes.Tensor):
        return nunes.maxpool_2d(x, self.kernel_size, self.stride)

    def __additional_repr__(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}"


class Conv2D(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _int2d = (3, 3),
            stride: _int2d = (1, 1),
            padding: _int2d = 0,
            bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = nutils.to_pair(kernel_size)
        self.stride = nutils.to_pair(stride)
        self.padding = nutils.to_pair(padding)

        self.weight = nunes.uniform((out_channels, in_channels, *self.kernel_size), low=-0.01, high=0.01).as_parameter()
        self.biases = nunes.uniform((out_channels,), low=-0.01, high=0.01).as_parameter() if bias else None

    def forward(self, x: nunes.Tensor) -> nunes.Tensor:
        return nunes.conv_2d(x, self.weight, self.biases, self.stride, self.padding)

    def __additional_repr__(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.biases is not None}"


__all__ = [
    "MaxPool2D",
    "Conv2D"
]
