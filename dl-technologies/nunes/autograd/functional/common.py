import typing as tp

import numpy as np

from .core import (BinaryFunction, Function, GradientType, PointwiseFunction,
                   ReduceFunction)


class Negate(PointwiseFunction):
    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        return [-output * np.ones(self.tensor.shape)]

    def __apply__(self) -> np.ndarray:
        return -self.tensor.numpy()


class IndicesSlice(PointwiseFunction):
    def __init__(self, tensor, *slices):
        super().__init__(tensor)
        self.slices = slices

    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        grad = np.zeros(self.tensor.shape)
        # print(output)
        grad.__setitem__(*self.slices, output)
        return [grad]

    def __apply__(self) -> np.ndarray:
        return self.tensor.numpy().__getitem__(*self.slices)


class Dropout(PointwiseFunction):
    def __init__(self, tensor, p: float = 0.5):
        super().__init__(tensor)
        self.p = p
        self.mask = None

    def __apply__(self) -> np.ndarray:
        rand = np.random.rand(*self.tensor.shape)
        self.mask = np.where(rand < self.p, 0, 1)
        return self.mask * self.tensor.numpy()

    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        return [output * self.mask]


class Reshape(PointwiseFunction):
    def __init__(self, tensor, shape: tp.Tuple[int, ...]):
        super().__init__(tensor)
        self.shape = shape

    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        return [output.reshape(self.tensor.shape)]

    def __apply__(self) -> np.ndarray:
        return self.tensor.numpy().reshape(self.shape)


class Pow(PointwiseFunction):
    def __init__(self, tensor, times: int):
        super().__init__(tensor)
        self.times = times

    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        return [output * self.times * self.tensor.numpy() ** (self.times - 1)]

    def __apply__(self) -> np.ndarray:
        return self.tensor.numpy() ** self.times


class Log(PointwiseFunction):
    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        return [output / self.tensor.numpy()]

    def __apply__(self) -> np.ndarray:
        return np.log(self.tensor.numpy())


class Exp(PointwiseFunction):
    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        return [output * self.tensor.numpy()]

    def __apply__(self) -> np.ndarray:
        return np.exp(self.tensor.numpy())


class Mean(ReduceFunction):
    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        return [output * np.ones(self.tensor.shape) / self.tensor.numel()]

    def __apply__(self) -> np.ndarray:
        return np.mean(self.tensor.numpy())


class Sum(ReduceFunction):
    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        return [output * np.ones(self.tensor.shape)]

    def __apply__(self) -> np.ndarray:
        return np.sum(self.tensor.numpy())


class Add(BinaryFunction):
    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        return [output * np.ones(self.lhs.shape), output * np.ones(self.rhs.shape)]

    def __apply__(self) -> np.ndarray:
        return self.lhs.numpy() + self.rhs.numpy()


class Sub(BinaryFunction):
    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        return [output * np.ones(self.lhs.shape), -output * np.ones(self.rhs.shape)]

    def __apply__(self) -> np.ndarray:
        return self.lhs.numpy() - self.rhs.numpy()


class Mul(BinaryFunction):
    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        return [output * self.rhs.numpy(), output * self.lhs.numpy()]

    def __apply__(self) -> np.ndarray:
        return self.lhs.numpy() * self.rhs.numpy()


class Div(BinaryFunction):
    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        return [output / self.rhs.numpy(), -output * self.lhs.numpy() / self.rhs.numpy() ** 2]

    def __apply__(self) -> np.ndarray:
        return self.lhs.numpy() / self.rhs.numpy()


class MatMul(BinaryFunction):
    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        return [output @ self.rhs.numpy().T, self.lhs.numpy().T @ output]

    def __apply__(self) -> np.ndarray:
        return self.lhs.numpy() @ self.rhs.numpy()


class Linear(Function):
    def __init__(self, x, weight, biases):
        super().__init__(graph=x.graph or weight.graph or biases.graph)
        self.x = x
        self.weight = weight
        self.biases = biases

    def __apply__(self) -> np.ndarray:
        mm = self.x.numpy() @ self.weight.numpy()
        return mm + self.biases.numpy() if self.biases else mm

    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        x, w = self.x.numpy(), self.weight.numpy()

        grad_x = output @ w.T
        grad_w = x.T @ output
        grad_b = None if self.biases is None else output.sum(axis=0)

        return [grad_x, grad_w, grad_b]

    def operands(self) -> tp.List:
        return [self.x, self.weight, self.biases]


__all__ = [
    "Negate",
    "IndicesSlice",
    "Dropout",
    "Reshape",
    "Pow",
    "Log",
    "Exp",
    "Mean",
    "Sum",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "MatMul",
    "Linear"
]
