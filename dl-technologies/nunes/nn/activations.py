import nunes

from .module import Module


class LeakyReLU(Module):
    def __init__(self, slope: float = 1e-2):
        super().__init__()
        self.slope = slope

    def forward(self, x: nunes.Tensor) -> nunes.Tensor:
        return nunes.leaky_relu(x, slope=self.slope)

    def __additional_repr__(self) -> str:
        return f"slope={self.slope}"


class ReLU(Module):
    def forward(self, x: nunes.Tensor) -> nunes.Tensor:
        return nunes.relu(x)


class Sigmoid(Module):
    def forward(self, x: nunes.Tensor) -> nunes.Tensor:
        return nunes.sigmoid(x)


class Softmax(Module):
    def forward(self, x: nunes.Tensor) -> nunes.Tensor:
        return nunes.softmax(x)


class LogSoftmax(Module):
    def forward(self, x: nunes.Tensor) -> nunes.Tensor:
        return nunes.log_softmax(x)


__all__ = [
    "LeakyReLU",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "LogSoftmax"
]
