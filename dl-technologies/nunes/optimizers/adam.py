import typing as tp

import numpy as np

from .core import Optimizer, ParametersGenerator


class Adam(Optimizer):
    """Adam optimizer"""
    def __init__(
            self,
            parameters: ParametersGenerator,
            lr: float = 1e-3,
            betas: tp.Tuple[float, float] = (0.99, 0.999),
            weight_decay: float = 0.0,
            eps: float = 1e-8
    ):
        super().__init__(parameters)
        self.lr = lr
        self.l2 = weight_decay
        self.betas = betas
        self.eps = eps

        self.storage = {}

    def step(self):
        b1, b2 = self.betas

        for i, param in enumerate(self.parameters):
            assert param.grad is not None, "Gradient is not set. Please use `forward` method first"

            grad = param.grad

            if i not in self.storage:
                self.storage[i] = (grad, grad ** 2)

            average, squared = self.storage[i]

            average = b1 * average + (1 - b1) * grad
            squared = b2 * squared + (1 - b2) * grad ** 2

            self.storage[i] = (average, squared)

            gradient = self.lr * (average / np.sqrt(squared + self.eps) + self.l2 * param.data)
            param.data -= gradient


__all__ = [
    "Adam"
]
