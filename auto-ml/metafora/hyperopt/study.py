from typing import Callable, Optional

import numpy as np
from tqdm import tqdm

from .optimizers import BayesianOptimizer
from .optimizers.optimizer import Optimizer
from .params import HyperParameters, TParamValues
from .surrogate.functions import SurrogateEstimator


class Study:
    def __init__(
            self,
            hyperparams: HyperParameters,
            optimizer: Optimizer = BayesianOptimizer()
    ):
        self.hyperparams = hyperparams
        self.__optimizer = optimizer.set_hyperparams(self.hyperparams)

        self.losses = []
        self.params = []

    def optimize(
            self,
            objective: Callable[[TParamValues], float],
            episodes: int,
            verbose: bool = False
    ):

        progress = range(episodes)
        if verbose:
            progress = tqdm(progress, desc="optimizing")

        for iteration in progress:

            losses, params = self.__optimizer.step(
                objective=objective,
                losses=self.losses,
                params=self.params,
                iteration=iteration
            )

            self.losses += losses
            self.params += params

            if verbose:
                progress.set_postfix_str(f"loss: {np.array(losses).mean():.4f}")

        self.__optimizer.step(objective, self.losses, self.params, iteration=episodes + 1)

    @property
    def best_score(self) -> float:
        return self.__optimizer.best_score

    @property
    def best_step(self) -> Optional[int]:
        return self.__optimizer.best_step

    @property
    def best_params(self) -> Optional[TParamValues]:
        return self.__optimizer.best_param

    @property
    def surrogate(self) -> Optional[SurrogateEstimator]:
        if hasattr(self.__optimizer, "surrogate"):
            return self.__optimizer.surrogate

    def __repr__(self):
        return f"{self.__class__.__name__}(\n\t{self.hyperparams}\n\n\toptimizer: {self.__optimizer}\n)"
