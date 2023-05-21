import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation: tp.Callable[[torch.Tensor], torch.Tensor]
    ):
        super(DenseBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = activation
        self._init_weights()

    def _init_weights(self):
        self.fc.weight.data.normal_(0, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.fc(x))


class DenseNet(nn.Module):
    def __init__(
            self,
            arch: tp.List[int],
            activation: tp.Callable[[torch.Tensor], torch.Tensor] = F.relu
    ):
        super(DenseNet, self).__init__()
        self.layers = nn.ModuleList([
            DenseBlock(
                in_features,
                out_features,
                activation
            ) for in_features, out_features in zip(arch, arch[1:])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


__all__ = [
    "DenseNet"
]
