from typing import Callable, List

from ..params import TParamValues
from .optimizer import Optimizer


class RandomOptimizer(Optimizer):
    def step(
            self,
            objective: Callable[[TParamValues], float],
            losses: List[float],
            params: List[TParamValues],
            iteration: int,
    ):
        param = self.hyperparams.sample()
        loss = objective(param.copy())

        if loss < self.best_score:
            self.best_step = iteration
            self.best_score = loss
            self.best_param = param

        return [loss], [param]


__all__ = [
    "RandomOptimizer"
]
