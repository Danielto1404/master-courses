import abc
import copy
import warnings
from typing import Optional, Union

import pandas as pd

from metafora.meta.types import ColumnsInputT, MetaFeaturesRawT


class FitTransform(abc.ABC):
    def __init__(self):
        self.is_fitted = False

    def fit(self, *args, **kwargs) -> "FitTransform":
        if self.is_fitted:
            warnings.warn("Already fitted.")
            return self

        self.is_fitted = True
        self._fit(*args, **kwargs)
        return self

    def transform(self, *args, **kwargs):
        assert self.is_fitted, "Please call `fit` method first before applying `transform` method"
        return self._transform(*args, **kwargs)

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)

    @abc.abstractmethod
    def _fit(self, *args, **kwargs):
        raise NotImplementedError("`_fit` method must be implemented in subclass")

    @abc.abstractmethod
    def _transform(self, *args, **kwargs):
        raise NotImplementedError("`_transform` method must be implemented in subclass")

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return repr(self)


class MetaFeaturesExtractor(FitTransform, abc.ABC):
    def __init__(self):
        super(MetaFeaturesExtractor, self).__init__()
        self.features: MetaFeaturesRawT = []

    def _aggregate_result(
            self,
            df: pd.DataFrame,
            return_df: bool = False,
            inplace: bool = False
    ) -> Union[pd.DataFrame, "MetaFeatures"]:
        features = MetaFeatures(self.features)

        if not return_df:
            return features

        return features.apply(df, inplace)

    def _transform(self, *args, **kwargs):
        return self._aggregate_result(*args, **kwargs)


class MetaFeaturesColumnsExtractor(MetaFeaturesExtractor, abc.ABC):
    def __init__(self, columns: ColumnsInputT):
        super(MetaFeaturesColumnsExtractor, self).__init__()

        if isinstance(columns, str):
            self.columns = [columns]
        else:
            self.columns = columns

        self.is_empty = len(self.columns) == 0

    def __repr__(self):
        return f"{self.__class__.__name__}(columns={self.columns})"


class MetaFeatures:
    def __init__(self, features: Optional[MetaFeaturesRawT] = None):

        if features is None:
            self.features = []
            self.columns = []
            self.values = []

        else:
            self.columns, self.values = map(list, zip(*features))
            self.features = features

    def apply(self, df: pd.DataFrame, inplace: bool = False):
        """
        Applies stored transformations for given DataFrame.

        :param df: pandas DataFrame
        :param inplace: whether transformation applied with given instance or copy
        :return: df
        """
        if not inplace:
            df = copy.deepcopy(df)

        df[self.columns] = self.values

        return df

    def concat(self, other: "MetaFeatures") -> "MetaFeatures":
        """
        Concatenates transformations.

        :param other: Transformation
        :return: new Transformation pipe which include both this and other transformation methods
        """
        return MetaFeatures(self.features + other.features)

    def __repr__(self):
        features = "\n\t".join(map(str, self.features))

        return f"MetaFeatures(\n\t{features}\n)"

    def __len__(self):
        return len(self.features)


__all__ = [
    "FitTransform",
    "MetaFeaturesExtractor",
    "MetaFeaturesColumnsExtractor",
    "MetaFeatures"
]
