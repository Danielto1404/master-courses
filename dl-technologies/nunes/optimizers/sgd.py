from .core import Optimizer, ParametersGenerator


class SGD(Optimizer):
    """SGD optimizer"""
    def __init__(self, parameters: ParametersGenerator, lr: float = 1e-4, weight_decay: float = 0.0):
        super().__init__(parameters)
        self.lr = lr
        self.l2 = weight_decay

    def step(self):
        for param in self.parameters:
            if param.requires_grad:
                gradient = self.lr * (param.grad + self.l2 * param.data)
                param.data -= gradient


__all__ = [
    "SGD"
]
