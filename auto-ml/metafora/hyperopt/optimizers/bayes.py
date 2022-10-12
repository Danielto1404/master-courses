from typing import Callable, List, Tuple

import numpy as np

from ..params import TParamValues
from ..surrogate.functions import GaussianProcessEstimator, SurrogateEstimator
from .optimizer import Optimizer


class BayesianOptimizer(Optimizer):
    def __init__(
            self,
            initial_samples: int = 5,
            population_size: int = 5,
            surrogate: SurrogateEstimator = GaussianProcessEstimator()
    ):
        super().__init__()

        self.initial_samples = initial_samples
        self.population_size = population_size
        self.surrogate = surrogate
        self.is_sampled = False

    def step(
            self,
            objective: Callable[[TParamValues], float],
            losses: List[float],
            params: List[TParamValues],
            iteration: int
    ):
        if not self.is_sampled:
            losses, params = self._make_initial_dataset(objective)
            index = np.array(losses).argmin()

            self.best_param = params[index]
            self.best_score = losses[index]

            vectorized = self.hyperparams.vectorize(params)
            self.surrogate.fit(vectorized, losses)

            self.is_sampled = True
            return losses, params

        # 1. find best params which minimize the acquisition function
        param = self._best_hyperparams_step()

        # 2. evaluate objective
        loss = objective(param)

        if loss < self.best_score:
            self.best_step = iteration
            self.best_score = loss
            self.best_param = param

        # 3. fit surrogate function
        vectorized = self.hyperparams.vectorize(params)
        self.surrogate.fit(vectorized, losses)

        return [loss], [param]

    def _make_initial_dataset(
            self,
            objective: Callable[[TParamValues], float]
    ) -> Tuple[List[float], List[TParamValues]]:
        losses, params = [], []
        for _ in range(self.initial_samples):
            param = self.hyperparams.sample()
            loss = objective(param)

            losses.append(loss)
            params.append(param)

        return losses, params

    def _best_hyperparams_step(self) -> TParamValues:
        samples = [
            self.hyperparams.cross(self.best_param, vectorize=True)
            for _ in range(self.population_size)
        ]

        params, xs = zip(*samples)

        scores = self.surrogate.estimate(
            x=np.array(xs),
            best_score=self.best_score
        )

        index = scores.argmax()

        return params[index].copy()

    def __repr__(self):
        return f"{self.__class__.__name__}(improvement={self.surrogate.improvement})"


__all__ = [
    "BayesianOptimizer"
]
