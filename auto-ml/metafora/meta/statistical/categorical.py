from typing import Callable, List

import numpy as np
import pandas as pd
import scipy.stats as stats

from metafora.meta.types import ColumnsInputT
from metafora.meta.utils import parse_str_from_enum

from ..core import MetaFeaturesColumnsExtractor
from ..variants import *


def get_category_proba_aggregation(agg: str) -> Callable:
    if agg == CategoricalVariants.DistinctClasses.value:
        return len
    elif agg == CategoricalVariants.Entropy.value:
        return stats.entropy


class CategoricalExtractor(MetaFeaturesColumnsExtractor):
    def __init__(
            self,
            columns: ColumnsInputT,
            categorical_aggregators: List[TCategorical],
            post_aggregators: List[TNumeric]
    ):
        super(CategoricalExtractor, self).__init__(columns)
        self.categorical_aggregators = parse_str_from_enum(categorical_aggregators)
        self.post_aggregators = parse_str_from_enum(post_aggregators)

    def _fit(
            self,
            df: pd.DataFrame,
            *args,
            **kwargs
    ):
        if self.is_empty:
            self.features = [
                (f"{cat_agg}_{agg}", np.nan)
                for cat_agg in self.categorical_aggregators
                for agg in self.post_aggregators
            ]
            return

        probabilities = [df[c].value_counts() / len(df) for c in self.columns]

        for cat_agg in self.categorical_aggregators:
            f = get_category_proba_aggregation(cat_agg)

            probas = pd.Series([f(proba) for proba in probabilities]).apply(self.post_aggregators)

            self.features += list(
                zip(
                    map(lambda agg: f"{cat_agg}_{agg}", self.post_aggregators),
                    probas
                )
            )

    def __repr__(self):
        stages = [
            f"{cat_agg} => {post_agg}"
            for cat_agg in self.categorical_aggregators
            for post_agg in self.post_aggregators
        ]

        stages = "\n\t\t".join(stages)

        return f"{self.__class__.__name__}(\n\t\t{stages}\n\t)"


__all__ = [
    "CategoricalExtractor"
]
