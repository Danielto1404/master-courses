import pickle

import nunes

from ..tensor import utils as nutils


class Module:
    """Base class for constructing neural networks"""

    def __init__(self):
        self.training = True

    def train(self):
        """Switches submodules to train mode"""
        self.training = True
        for _, m in self.submodules():
            m.train()

        return self

    def eval(self):
        """Switches submodules to eval mode"""
        self.training = False
        for _, m in self.submodules():
            m.eval()

        return self

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "Module":
        with open(path, "rb") as f:
            return pickle.load(f)

    def forward(self, *args, **kwargs):
        return NotImplemented("`forward` method must be implemented in subclasses")

    def submodules(self):
        """Returns generator for all submodules with their names"""
        for p, m in vars(self).items():
            if isinstance(m, Module):
                yield p, m

    def parameters(self):
        """Return generator for all `Parameter` properties in each submodule"""
        for p in vars(self).values():
            if isinstance(p, Module):
                yield from p.parameters()

            if isinstance(p, nunes.Parameter):
                yield p

    def __call__(self, *args, **kwargs):
        """Alias for `.forward(*args, **kwargs)` method"""
        return self.forward(*args, **kwargs)

    def __additional_repr__(self) -> str:
        return ""

    def __repr_tree__(self, property_name: str | None = None, depth: int = 0):
        submodules = [m.__repr_tree__(p, depth + 1) for p, m in self.submodules()]

        property_name = "" if property_name is None else f"({property_name}): "

        if len(submodules) == 0:
            tree = f"{nutils.tab * depth}{property_name}{self.__class__.__name__}({self.__additional_repr__()})"
        else:
            tree = [
                f"{nutils.tab * depth}{property_name}{self.__class__.__name__}(",
                *submodules,
                f"{nutils.tab * depth})"
            ]
            tree = nutils.nl.join(tree)

        return tree

    def __repr__(self):
        return self.__repr_tree__()


__all__ = [
    "Module"
]
