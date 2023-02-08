import abc
import typing as tp

import nunes

ParametersGenerator = tp.Iterable[nunes.Parameter]


class Optimizer(abc.ABC):
    """Base class which handles gradient descent strategy"""

    def __init__(self, parameters: ParametersGenerator):
        if isinstance(parameters, list):
            self.parameters = parameters
        else:
            self.parameters = list(parameters)

    def zero_grad(self):
        """Resets gradients for given parameters"""
        for p in self.parameters:
            p.grad = None

    @abc.abstractmethod
    def step(self):
        """
        Applies optimization step. Use it only after `.backward` method.
        """
        return NotImplemented()

    def __repr__(self):
        return f"{self.__class__.__name__}()"


__all__ = [
    "Optimizer",
    "ParametersGenerator"
]
