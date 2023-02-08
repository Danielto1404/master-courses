import typing as tp

import numpy as np
import scipy.signal as signal

from ...tensor.utils import to_pair
from .core import Function, GradientType, PointwiseFunction


class MaxPool2D(PointwiseFunction):
    def __init__(
            self,
            tensor,
            kernel_size: tp.Union[int, tp.Tuple[int, int]] = (2, 2),
            stride: tp.Union[int, tp.Tuple[int, int]] = (2, 2)
    ):
        assert len(tensor.shape) == 4, f"Expected 4D tensor, but got: {len(self.tensor.shape)}D"
        super().__init__(tensor)

        self.batches, self.channels, self.h, self.w = tensor.shape

        self.sy, self.sx = to_pair(stride)
        self.ky, self.kx = to_pair(kernel_size)

        self.out_h = self.__output_size__(self.h, self.sy, self.ky)
        self.out_w = self.__output_size__(self.w, self.sx, self.kx)

        self.mask = np.zeros(self.output_shape, dtype=np.int)

    @property
    def output_shape(self):
        return self.batches, self.channels, self.out_h, self.out_w

    @staticmethod
    def __output_size__(size: int, stride: int, kernel_size: int) -> int:
        return (size - kernel_size) // stride + 1

    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        gradient = np.zeros(self.tensor.shape)

        for b in range(self.batches):
            for c in range(self.channels):
                for y in range(self.out_h):
                    for x in range(self.out_w):
                        argmax = self.mask[b, c, y, x]
                        dx, dy = np.unravel_index(argmax, (self.ky, self.kx))
                        gradient[b, c, self.sy * y + dx, self.sx * x + dy] += output[b, c, y, x]

        return [gradient]

    def __apply__(self) -> np.ndarray:
        batch = self.tensor.numpy()
        output = np.zeros(self.output_shape)

        for b in range(self.batches):
            for c in range(self.channels):
                for y in range(self.out_h):
                    for x in range(self.out_w):
                        region = batch[b, c, self.sy * y: self.sy * y + self.ky, self.sx * x: self.sx * x + self.kx]
                        output[b, c, y, x] = np.max(region)
                        self.mask[b, c, y, x] = np.argmax(region)

        return output


class Conv2D(Function):
    def __init__(self, tensor, kernel, biases, stride, padding):
        assert len(tensor.shape) == 4, f"Expected 4D input tensor, but got: {len(tensor.shape)}D"
        assert len(kernel.shape) == 4, f"Expected 4D kernel tensor, but got: {len(kernel.shape)}D"

        super().__init__(graph=tensor.graph or kernel.graph or biases.graph)
        self.tensor = tensor
        self.kernel = kernel
        self.biases = biases

        self.sy, self.sx = to_pair(stride)
        self.py, self.px = to_pair(padding)

        self.batches, self.in_channels, self.h, self.w = tensor.shape
        self.out_channels, _, self.ky, self.kx = kernel.shape

        self.out_h = self.__output_size__(self.h, self.ky, self.sy, self.py)
        self.out_w = self.__output_size__(self.w, self.kx, self.sx, self.px)

    @staticmethod
    def __output_size__(size: int, kernel_size: int, stride: int, padding: int) -> int:
        return (size + 2 * padding - kernel_size) // stride + 1

    @property
    def output_shape(self):
        return self.batches, self.out_channels, self.out_h, self.out_w

    def padded(self) -> np.ndarray:
        paddings = ((0, 0), (0, 0), (self.py, self.py), (self.px, self.px))
        tensor = np.pad(self.tensor.numpy(), paddings, constant_values=0)

        return tensor

    def __backward__(self, output: GradientType) -> tp.List[GradientType]:
        """
        :param output: numpy.ndarray of shape: (batches, out_channels, height, width)
        """
        tensor = self.padded()
        kernel = self.kernel.numpy()

        grad_x = np.zeros_like(tensor)
        grad_k = np.zeros((self.out_channels, self.in_channels, self.ky, self.kx))
        grad_b = output.sum(axis=(0, 2, 3)) if self.biases else None

        for batch in range(self.batches):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for y in range(self.out_h):
                        for x in range(self.out_w):
                            x0 = x * self.sx
                            x1 = x0 + self.kx
                            y0 = y * self.sy
                            y1 = y0 + self.ky

                            grad = output[batch, out_c, y, x]

                            grad_x[batch, in_c, y0: y1, x0: x1] += grad * kernel[out_c, in_c]

                            region = tensor[batch, in_c, y0: y1, x0: x1]
                            grad_k[out_c, in_c] += region * grad

        if self.px != 0 and self.py != 0:
            grad_x = grad_x[..., self.py:-self.py, self.px:-self.px]
        elif self.px != 0:
            grad_x = grad_x[..., self.px:-self.px]
        elif self.py != 0:
            grad_x = grad_x[..., self.py:-self.py, :]

        return [grad_x, grad_k, grad_b]

    def __apply__(self) -> np.ndarray:
        output = np.empty(self.output_shape)

        tensor = self.padded()
        kernel = self.kernel.numpy()
        biases = self.biases.numpy() if self.biases else np.zeros(self.out_channels)

        for batch in range(self.batches):
            for out_c in range(self.out_channels):
                conv = signal.correlate(tensor[batch], kernel[out_c], mode="valid")
                conv = conv[0, ::self.sy, ::self.sx]

                output[batch, out_c] = conv + biases[out_c]

        return output

    def operands(self) -> tp.List:
        return [self.tensor, self.kernel, self.biases]


__all__ = [
    "MaxPool2D",
    "Conv2D"
]
