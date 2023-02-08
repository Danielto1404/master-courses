import nunes

from ..module import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nunes.uniform((in_features, out_features), -0.1, 0.1).as_parameter()
        self.biases = nunes.uniform((out_features,), -0.1, 0.1).as_parameter() if bias else None

    def forward(self, x: nunes.Tensor) -> nunes.Tensor:
        return nunes.linear(x, weight=self.weight, biases=self.biases)

    def __additional_repr__(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.biases is not None}"


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0 < p < 1, "dropout probability must be in (0, 1) range"
        self.p = p

    def forward(self, x: nunes.Tensor) -> nunes.Tensor:
        if self.training:
            x = nunes.dropout(x, p=self.p)
            scale = 1 / (1 - self.p)
            return x * scale
        else:
            return x

    def __additional_repr__(self) -> str:
        return f"p={self.p}"


class Flatten(Module):

    def forward(self, x: nunes.Tensor) -> nunes.Tensor:
        batch_size = x.shape[0]
        return nunes.reshape(x, (batch_size, -1))


class Reshape(Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: nunes.Tensor) -> nunes.Tensor:
        return nunes.reshape(x, self.shape)

    def __additional_repr__(self) -> str:
        return f"shape={self.shape}"


__all__ = [
    "Linear",
    "Dropout",
    "Flatten",
    "Reshape"
]
