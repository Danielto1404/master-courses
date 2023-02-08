import abc

import numpy as np

import nunes


class Loss(abc.ABC):
    """Base loss class"""
    __reductions__ = ["mean", "sum"]

    def __init__(self, reduction: str = "mean"):
        assert reduction in self.__reductions__, \
            f"Unknown reduction type: {reduction}, please use of this: {self.__reductions__}"

        self.reduction = reduction

    @abc.abstractmethod
    def forward(self, output: nunes.Tensor, target: nunes.Tensor) -> nunes.Tensor:
        """
        Aggregates loss on each object depending on provided reduction type.
        If `output` tensor `requires_grad` property equals to `True` then `.backward()`
        method can be used for gradients calculation.

        :param output: tensor with predicted values
        :param target: tensor with target values

        :return: loss tensor
        """
        return NotImplemented()

    def reduce_output(self, output: nunes.Tensor) -> nunes.Tensor:
        if self.reduction == "mean":
            return output.mean()
        elif self.reduction == "sum":
            return output.sum()
        else:
            raise NotImplementedError(f"Unknown reduction type: {self.reduction}")

    def __call__(self, output: nunes.Tensor, target: nunes.Tensor) -> nunes.Tensor:
        return self.forward(output, target)

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction={self.reduction})"


class MSELoss(Loss):
    """Implementation of MSE loss"""

    def forward(self, output: nunes.Tensor, target: nunes.Tensor) -> nunes.Tensor:
        """Calculates mean squared error loss"""
        assert output.shape == target.shape, f"Shapes missmatch: expected: {target.shape}, got: {output.shape}"
        mse = (output - target) ** 2
        return self.reduce_output(mse)


class NLLLoss(Loss):
    """Implementation of Negative Log Likelihood loss"""

    def forward(self, probas: nunes.Tensor, target: nunes.Tensor) -> nunes.Tensor:
        """Calculates negative log likelihood loss"""
        assert len(probas.shape) == 2, "output tensor must be of size (batch, classes)"
        assert len(target.shape) == 1, "target tensor must be of size (batch,)"

        b = np.arange(len(target))
        t = target.numpy()

        nll = -probas[b, t].log()
        return self.reduce_output(nll)


class CrossEntropyLoss(NLLLoss):
    """Implementation of Cross Entropy loss"""

    def forward(self, output: nunes.Tensor, target: nunes.Tensor) -> nunes.Tensor:
        """Calculates cross entropy loss"""
        probas = nunes.softmax(output)
        return super().forward(probas, target)


class BCELoss(Loss):
    """Implementation of Binary Cross Entropy loss"""

    def forward(self, output: nunes.Tensor, target: nunes.Tensor) -> nunes.Tensor:
        assert len(output.shape) == 1, "output tensor must be of size (batch, )"
        assert len(target.shape) == 1, "target tensor must be of size (batch, )"
        assert output.shape == target.shape, \
            f"output shape doesn't match with target shape. expected: {target.shape}, got: {output.shape}."

        bce = -target * nunes.log(output) + (target - 1) * nunes.log(-output + 1)
        return self.reduce_output(bce)


class BCEWithLogits(BCELoss):
    """Implementation of Binary Cross Entropy loss where last layer with sigmoid activation can be skipped"""

    def forward(self, output: nunes.Tensor, target: nunes.Tensor) -> nunes.Tensor:
        logits = nunes.sigmoid(output)
        return super().forward(logits, target)


__all__ = [
    "Loss",
    "MSELoss",
    "CrossEntropyLoss",
    "BCELoss",
    "BCEWithLogits"
]
