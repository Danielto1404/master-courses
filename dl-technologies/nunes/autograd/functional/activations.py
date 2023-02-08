import typing as tp

import numpy as np

from .core import GradientType, PointwiseFunction


class LeakyReLU(PointwiseFunction):
    def __init__(self, tensor, slope: float):
        super().__init__(tensor=tensor)
        self.slope = slope
        self.negative_idx = None

    def __apply__(self) -> np.ndarray:
        self.negative_idx = self.tensor.numpy() <= 0
        x = self.tensor.numpy()
        return np.where(self.negative_idx, self.slope * x, x)

    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        grad = np.ones(self.tensor.shape)
        grad[self.negative_idx] = self.slope
        return [output * grad]


class ReLU(LeakyReLU):
    def __init__(self, tensor):
        super().__init__(tensor, slope=0.0)


class Sigmoid(PointwiseFunction):
    def __init__(self, tensor):
        super().__init__(tensor)

    def __apply__(self) -> np.ndarray:
        return 1 / (1 + np.exp(-self.tensor.numpy()))

    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        sigmoid = self.result.numpy()
        return [output * sigmoid * (1 - sigmoid)]


class Softmax(PointwiseFunction):
    def __apply__(self) -> np.ndarray:
        exps = np.exp(self.tensor.numpy())
        norm = exps.sum(axis=1, keepdims=True)

        return exps / norm

    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        softmax = self.result.numpy()
        shift = (output * softmax).sum(axis=1, keepdims=True)
        return [softmax * (output - shift)]


class LogSoftmax(PointwiseFunction):
    def __apply__(self) -> np.ndarray:
        exps = np.exp(self.tensor.numpy())
        norm = exps.sum(axis=1, keepdims=True)

        self.softmax = exps / norm

        return np.log(self.softmax)

    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        shift = (output * self.softmax).sum(axis=1, keepdims=True)
        return [output - shift]


__all__ = [
    "LeakyReLU",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "LogSoftmax"
]
