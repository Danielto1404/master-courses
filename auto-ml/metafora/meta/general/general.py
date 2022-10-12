from typing import List

import numpy as np
import pandas as pd

from metafora.meta.types import ColumnsInputT
from metafora.meta.utils import parse_str_from_enum

from ..core import *
from ..variants import TNumeric


class ShapeExtractor(MetaFeaturesExtractor):
    def _fit(
            self,
            df: pd.DataFrame,
            *args,
            **kwargs
    ):
        n_objects, n_features = df.shape
        self.features = [("features_objects_rate", n_features / n_objects)]


class ClassesNumberExtractor(MetaFeaturesColumnsExtractor):
    def __init__(self, columns: ColumnsInputT, post_aggregators: List[TNumeric]):
        super(ClassesNumberExtractor, self).__init__(columns)
        self.post_aggregators = parse_str_from_enum(post_aggregators)

    def _fit(
            self,
            df: pd.DataFrame,
            *args,
            **kwargs
    ):
        if self.is_empty:
            self.features = [(f"unique_{stat}", np.nan) for stat in self.post_aggregators]
            return

        uniques = pd.Series([len(df[c].unique()) / len(df) for c in self.columns])
        uniques = uniques.apply(self.post_aggregators)

        self.features = [(f"unique_{stat}", value) for stat, value in zip(self.post_aggregators, uniques)]


class SparsityExtractor(MetaFeaturesExtractor):

    def __init__(self, post_aggregators: List[TNumeric]):
        super(SparsityExtractor, self).__init__()
        self.post_aggregators = parse_str_from_enum(post_aggregators)

    def _fit(self, df: pd.DataFrame, *args, **kwargs):
        na_stats = df.isna().mean(0).apply(self.post_aggregators)

        self.features = [(f"sparsity_{stat}", value) for stat, value in zip(self.post_aggregators, na_stats)]


class CatNumColumnsRateExtractor(MetaFeaturesExtractor):
    @staticmethod
    def is_category(dtype: str) -> bool:
        dtype = str(dtype)
        return not (dtype.startswith("int") or dtype.startswith("float"))

    def _fit(
            self,
            df: pd.DataFrame,
            *args,
            **kwargs
    ):
        _, num_features = df.shape
        categorical = list(filter(self.is_category, df.dtypes.values.tolist()))
        self.features = [
            ("cat_cols_rate", len(categorical) / num_features),
            ("num_cols_rate", 1 - len(categorical) / num_features)
        ]


__all__ = [
    "ShapeExtractor",
    "ClassesNumberExtractor",
    "SparsityExtractor",
    "CatNumColumnsRateExtractor"
]
