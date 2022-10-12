import abc
import random
from typing import Any, Dict, List, Tuple, Union

import numpy as np

TParamValues = Dict[str, Any]


class HyperParameter(abc.ABC):
    @abc.abstractmethod
    def sample(self):
        """
        Randomly sample value from parameter space
        """
        raise NotImplementedError("`sample` method must be implemented in subclass")

    @abc.abstractmethod
    def vectorize(self, value) -> np.array:
        """
        :param value: raw value
        :return: vectorized value
        """
        raise NotImplementedError("`vectorize` method must be implemented in subclass")

    @abc.abstractmethod
    def transform_vector(self, value):
        raise NotImplementedError("`transform_vector` method must be implemented in subclass")

    @abc.abstractmethod
    def single(self) -> bool:
        """
        Checks whether if parameter space contains single value
        """
        raise NotImplementedError("`single` method must be implemented in subclass")

    @property
    @abc.abstractmethod
    def shape(self) -> int:
        raise NotImplementedError("`shape` method must be implemented in subclass")

    @abc.abstractmethod
    def cross(self, value):
        raise NotImplementedError("`cross` method must be implemented in subclass")


class RangeParameter(HyperParameter, abc.ABC):
    def __init__(self, lower, upper, log_scale: bool = False):
        self.lower = lower
        self.upper = upper

        self.log_scale = log_scale

    def vectorize(self, value) -> np.ndarray:
        return np.array([value])

    def single(self) -> bool:
        return self.lower == self.upper

    @property
    def shape(self) -> int:
        return 1

    def __repr__(self):
        return f"{self.__class__.__name__}(lower={self.lower}, upper={self.upper}, log_scale={self.log_scale})"


class IntParameter(RangeParameter):
    def sample(self):
        if self.log_scale:
            power = np.random.uniform(np.log(self.lower), np.log(self.upper))
            value = np.exp(power)
            value = round(value)

            return value

        return random.randint(self.lower, self.upper)

    def transform_vector(self, value: np.ndarray):
        value = value[0]
        return np.clip(round(value), self.lower, self.upper)

    def cross(self, value):
        new = self.sample()
        return (new + value) // 2


class FloatParameter(RangeParameter):
    def sample(self):
        if self.log_scale:
            power = np.random.uniform(np.log(self.lower), np.log(self.upper))
            value = np.exp(power)

            return value

        return np.random.uniform(self.lower, self.upper)

    def transform_vector(self, value: np.ndarray):
        value = value[0]
        return np.clip(value, self.lower, self.upper)

    def cross(self, value):
        new = self.sample()
        return (new + value) / 2


class FixedParameter(HyperParameter):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def cross(self, _):
        return self.value

    def vectorize(self, value) -> np.array:
        # dummy zero
        return np.array([0])

    @property
    def shape(self) -> int:
        return 1

    def transform_vector(self, _):
        return self.value

    def single(self) -> bool:
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class CategoricalParameter(HyperParameter):
    def __init__(self, *variants):
        self.variants = list(variants)

    def sample(self):
        return random.choice(self.variants)

    def vectorize(self, value) -> np.ndarray:
        vector = np.zeros(len(self.variants))
        position = self.variants.index(value)
        vector[position] = 1.0
        return vector

    def transform_vector(self, value):
        index = value.argmax()
        return self.variants[index]

    def cross(self, value):
        if random.random() < 0.8:
            return value
        else:
            return self.sample()

    def single(self) -> bool:
        return len(self.variants) == 1

    @property
    def shape(self) -> int:
        return len(self.variants)

    def __repr__(self):
        variants = ", ".join(self.variants)
        return f"{self.__class__.__name__}({variants})"


class HyperParameters:
    def __init__(self, params: List[Tuple[str, HyperParameter]]):
        self.param_dict = dict(params)
        self.changeable = [name for name, p in self.param_dict.items() if not p.single()]
        self.fixed = [name for name, p in self.param_dict.items() if p.single()]

    @property
    def names(self):
        return self.param_dict.keys()

    @property
    def params(self):
        return self.param_dict.values()

    def sample(
            self,
            vectorize: bool = False
    ) -> Union[TParamValues, Tuple[TParamValues, np.ndarray]]:
        values = [p.sample() for p in self.params]
        params = dict(zip(self.names, values))

        return self._return_params(params, vectorize=vectorize)

    def vectorize(
            self,
            param: Union[TParamValues, List[TParamValues]]
    ) -> np.ndarray:
        def vectorize_param(p):
            return np.hstack(
                [self.param_dict[name].vectorize(p[name]) for name in self.changeable]
            )

        if isinstance(param, list):
            return [vectorize_param(p) for p in param]
        else:
            return vectorize_param(param)

    def transform_vector(self, vector: np.ndarray) -> TParamValues:

        params = {}
        index = 0

        for name in self.changeable:
            param = self.param_dict[name]
            value = param.transform_vector(vector[index:index + param.shape])
            index = index + param.shape

            params[name] = value

        for name in self.fixed:
            param = self[name]
            params[name] = param.sample()

        return params

    def shape(self):
        return np.sum([self[name].shape for name in self.changeable])

    def sample_vectors(self, n) -> np.ndarray:
        return np.random.rand(n, self.shape())

    def cross(
            self,
            params: TParamValues,
            vectorize: bool = False
    ) -> Union[TParamValues, Tuple[TParamValues, np.ndarray]]:

        crossed = {}

        for name, value in params.items():
            crossed[name] = self[name].cross(value)

        return self._return_params(crossed, vectorize)

    def _return_params(
            self,
            params: TParamValues,
            vectorize: bool
    ) -> Union[TParamValues, Tuple[TParamValues, np.ndarray]]:
        if vectorize:
            return params, self.vectorize(params)
        else:
            return params

    def __getitem__(self, param_name: str) -> HyperParameter:
        return self.param_dict[param_name]

    def __len__(self):
        return len(self.param_dict)

    def __repr__(self):
        width = max(map(len, self.names))
        params = "\n\t\t".join([f"{n: <{width}} : {p}" for n, p in self.param_dict.items()])
        return f"hyperparameters:\n\t\t{params}"


__all__ = [
    "HyperParameter",
    "IntParameter",
    "FloatParameter",
    "FixedParameter",
    "CategoricalParameter",
    "HyperParameters",
    "TParamValues"
]
