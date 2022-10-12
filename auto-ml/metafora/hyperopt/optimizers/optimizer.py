import abc
from typing import Callable, List, Optional, Tuple

from ..params import HyperParameters, TParamValues

_INF = float("inf")


class Optimizer(abc.ABC):
    """
    Abstract core class for objective optimization
    """

    def __init__(self):
        self.hyperparams: Optional[HyperParameters] = None
        self.best_param: Optional[TParamValues] = None
        self.best_score = _INF
        self.best_step = 0

    def set_hyperparams(self, hyperparams: HyperParameters):
        self.hyperparams = hyperparams
        return self

    @abc.abstractmethod
    def step(
            self,
            objective: Callable[[TParamValues], float],
            losses: List[float],
            params: List[TParamValues],
            iteration: int,
    ) -> Tuple[List[float], List[TParamValues]]:
        """
        Runs a given number of episodes for optimizing objective function

        :param objective: callable function which takes an Episode instances and returns error on given episode
        :param losses list of losses
        :param params list of hyperparameters
        :param iteration number of current iteration

        :return best score and params
        """
        raise NotImplementedError("`optimize` method must be implemented in subclass")

    def __repr__(self):
        return f"{self.__class__.__name__}()"


__all__ = [
    "Optimizer"
]
