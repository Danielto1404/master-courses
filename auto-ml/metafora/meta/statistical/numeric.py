from typing import List

import numpy as np
import pandas as pd

from metafora.meta.types import ColumnsInputT
from metafora.meta.utils import parse_str_from_enum

from ..core import MetaFeaturesColumnsExtractor
from ..variants import *


class NumericExtractor(MetaFeaturesColumnsExtractor):
    def __init__(
            self,
            columns: ColumnsInputT,
            column_aggregators: List[TNumeric],
            post_aggregators: List[TNumeric]
    ):
        super(NumericExtractor, self).__init__(columns)
        self.column_aggregators = parse_str_from_enum(column_aggregators)
        self.post_aggregators = parse_str_from_enum(post_aggregators)

    def _fit(
            self,
            df: pd.DataFrame,
            *args,
            **kwargs
    ):
        if self.is_empty:
            self.features = [
                (f"{col_agg}_{agg}", np.nan)
                for col_agg in self.column_aggregators
                for agg in self.post_aggregators
            ]
            return

        aggregations = df[self.columns] \
            .apply(self.column_aggregators) \
            .apply(self.post_aggregators, axis=1)

        names = [f"{col}_{post}" for col in self.column_aggregators for post in self.post_aggregators]
        values = aggregations.values.flatten()

        self.features = list(zip(names, values))

    def __repr__(self):
        stages = [
            f"{column_agg} => {post_agg}"
            for column_agg in self.column_aggregators
            for post_agg in self.post_aggregators
        ]

        stages = "\n\t\t".join(stages)

        return f"{self.__class__.__name__}(\n\t\t{stages}\n\t)"


__all__ = [
    "NumericExtractor"
]
