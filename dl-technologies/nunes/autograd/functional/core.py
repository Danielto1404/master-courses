import abc
import typing as tp

import numpy as np

GradientType = np.ndarray | int | float | None


class BackwardError(Exception):
    pass


class Function(abc.ABC):
    def __init__(
            self,
            graph: tp.Optional = None
    ):
        self.result: tp.Optional = None
        self.graph = graph

    @property
    def requires_grad(self) -> bool:
        return self.graph is not None

    @abc.abstractmethod
    def __apply__(self) -> np.ndarray:
        return NotImplemented()

    @abc.abstractmethod
    def __backward__(self, _: GradientType) -> tp.List[GradientType]:
        return NotImplemented()

    @abc.abstractmethod
    def operands(self) -> tp.List:
        """Returns list of operands that were used for calculation"""
        return NotImplemented()

    def backward(self, output: GradientType) -> tp.List[GradientType]:
        if self.result is None:
            raise BackwardError(
                "This operation is not evaluated yet. Please evaluate operation before calculating gradient."
            )

        return self.__backward__(output)

    def apply(self, tensor_builder: tp.Callable):
        self.result = tensor_builder(
            self.__apply__(),
            self.requires_grad
        )

        if self.requires_grad:
            self.graph.add(self)
            self.result.graph = self.graph

        return self.result

    def __call__(self, tensor_builder: tp.Callable):
        return self.apply(tensor_builder)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class PointwiseFunction(Function, abc.ABC):
    def __init__(self, tensor):
        super().__init__(graph=tensor.graph)
        self.tensor = tensor

    def operands(self):
        return [self.tensor]


class ReduceFunction(PointwiseFunction, abc.ABC):
    pass


class BinaryFunction(Function, abc.ABC):
    def __init__(self, lhs, rhs):
        super().__init__(graph=lhs.graph or rhs.graph)
        self.lhs = lhs
        self.rhs = rhs

    def operands(self):
        return [self.lhs, self.rhs]


__all__ = [
    "Function",
    "PointwiseFunction",
    "ReduceFunction",
    "BinaryFunction",
    "GradientType",
    "BackwardError"
]
