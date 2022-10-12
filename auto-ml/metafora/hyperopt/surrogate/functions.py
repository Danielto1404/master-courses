import abc

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from .acquisitions import ConfidenceBound, Improvement


class SurrogateEstimator(abc.ABC):

    def __init__(
            self,
            improvement: Improvement = ConfidenceBound(),
            normalize_x: bool = False
    ):
        self.improvement = improvement
        self.normalize_x = normalize_x
        self.scaler = None

    def _fit_x(self, x):
        if self.normalize_x:
            self.scaler = StandardScaler()
            x = self.scaler.fit_transform(x)

        return x

    def _transform_x(self, x):
        if self.scaler is not None:
            x = self.scaler.transform(x)

        return x

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
        """
        Approximates surrogate function

        :param x: meta-dataset of hyperparameters
        :param y: real objective function value
        """
        raise NotImplementedError("`fit` method must be implemented in subclass")

    @abc.abstractmethod
    def predict(self, x: np.ndarray, return_std: bool = False, **kwargs) -> np.ndarray[float]:
        raise NotImplementedError("`predict` method must be implemented in subclass")

    @abc.abstractmethod
    def estimate(self, x: np.ndarray, **kwargs) -> np.ndarray[float]:
        """
        Estimates surrogate function mean and std
        :param x: hyperparameters meta-dataset
        :return: improvement function scores
        """
        raise NotImplementedError("`estimate` method must be implemented in subclass")

    def _calc_improvements(self, mus: np.ndarray, sigmas: np.ndarray, **kwargs) -> np.ndarray[float]:
        return np.array(
            [self.improvement.score(mu, sigma, **kwargs) for mu, sigma in zip(mus, sigmas)]
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(improvement={self.improvement})"


class SklearnSurrogateEstimatorMixin(SurrogateEstimator, abc.ABC):
    """
    Surrogate estimator wrapper which approximates surrogate function with sklearn model
    """

    def __init__(self, model, improvement: Improvement = ConfidenceBound(), normalize_x: bool = False):
        super().__init__(improvement, normalize_x)
        self.model = model

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
        """
        Approximates surrogate function with given sklearn model

        :param x: meta-dataset of hyperparameters
        :param y: real objective function value
        """
        x = super()._fit_x(x)
        self.model.fit(x, y)

    def estimate(self, x: np.ndarray, **kwargs) -> np.ndarray[float]:
        mus, sigmas = self.predict(x, return_std=True)
        return self._calc_improvements(mus, sigmas, **kwargs)

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)
        return self


class RandomForestEstimator(SklearnSurrogateEstimatorMixin):
    def __init__(
            self,
            improvement: Improvement = ConfidenceBound(),
            normalize_x: bool = False,
            **kwargs
    ):
        model = RandomForestRegressor(**kwargs)
        super().__init__(model, improvement, normalize_x)

    def predict(self, x: np.ndarray, return_std: bool = False, **kwargs) -> np.ndarray[float]:
        x = super()._transform_x(x)

        if return_std:
            predictions = [tree.predict(x) for tree in self.model.estimators_]
            predictions = np.array(predictions)

            mus, sigmas = predictions.mean(axis=0), predictions.std(axis=0)

            return mus, sigmas

        return self.model.predict(x)


class GaussianProcessEstimator(SklearnSurrogateEstimatorMixin):
    def __init__(
            self,
            improvement: Improvement = ConfidenceBound(),
            normalize_x: bool = False,
            **kwargs
    ):
        model = GaussianProcessRegressor(**kwargs)
        super().__init__(model, improvement, normalize_x)

    def predict(self, x: np.ndarray, return_std: bool = False, **kwargs) -> np.ndarray[float]:
        x = super()._transform_x(x)
        return self.model.predict(x, return_std=return_std)


__all__ = [
    "SurrogateEstimator",
    "RandomForestEstimator",
    "GaussianProcessEstimator"
]
