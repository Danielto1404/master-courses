import typing as tp

from nunes.autograd.functional import *

from .tensor import Tensor, tensor


def exp(t: Tensor) -> Tensor:
    return Exp(t).apply(tensor)


def sum(t: Tensor) -> Tensor:
    return Sum(t).apply(tensor)


def mean(t: Tensor) -> Tensor:
    return Mean(t).apply(tensor)


def log(t: tensor) -> Tensor:
    return Log(t).apply(tensor)


def reshape(t: Tensor, shape: tp.Tuple[int, ...]) -> Tensor:
    return Reshape(t, shape).apply(tensor)


def add(a: Tensor, b: Tensor) -> Tensor:
    return Add(a, b).apply(tensor)


def div(a: Tensor, b: Tensor) -> Tensor:
    return Div(a, b).apply(tensor)


def mul(a: Tensor, b: Tensor) -> Tensor:
    return Mul(a, b).apply(tensor)


def sub(a: Tensor, b: Tensor) -> Tensor:
    return Sub(a, b).apply(tensor)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatMul(a, b).apply(tensor)


def indices_slice(t: Tensor, *slices) -> Tensor:
    return IndicesSlice(t, *slices).apply(tensor)


def negate(t: Tensor) -> Tensor:
    return Negate(t).apply(tensor)


def pow(t: Tensor, times) -> Tensor:
    return Pow(t, times).apply(tensor)


def square(t: Tensor) -> Tensor:
    return pow(t, times=2)


def leaky_relu(t: Tensor, slope: float = 1e-3) -> Tensor:
    return LeakyReLU(t, slope=slope).apply(tensor)


def relu(t: Tensor) -> Tensor:
    return ReLU(t).apply(tensor)


def sigmoid(t: Tensor) -> Tensor:
    return Sigmoid(t).apply(tensor)


def softmax(t: Tensor) -> Tensor:
    return Softmax(t).apply(tensor)

def log_softmax(t: Tensor) -> Tensor:
    return LogSoftmax(t).apply(tensor)


def conv_2d(t: Tensor, kernel: Tensor, biases: tp.Optional[Tensor], stride, padding) -> Tensor:
    return Conv2D(t, kernel=kernel, biases=biases, stride=stride, padding=padding).apply(tensor)


def maxpool_2d(t: Tensor, kernel_size=(2, 2), stride=(2, 2)) -> Tensor:
    return MaxPool2D(t, kernel_size=kernel_size, stride=stride).apply(tensor)


def dropout(t: Tensor, p: float = 0.5) -> Tensor:
    return Dropout(t, p=p).apply(tensor)


def linear(x: Tensor, weight: Tensor, biases: tp.Optional[Tensor]) -> Tensor:
    return Linear(x, weight, biases).apply(tensor)


__all__ = [
    "exp",
    "sum",
    "mean",
    "log",
    "reshape",
    "add",
    "div",
    "mul",
    "sub",
    "matmul",
    "indices_slice",
    "negate",
    "pow",
    "square",
    "leaky_relu",
    "relu",
    "sigmoid",
    "softmax",
    "log_softmax",
    "conv_2d",
    "maxpool_2d",
    "dropout",
    "linear"
]
