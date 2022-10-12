from typing import List

import numpy as np
import pandas as pd

from metafora.meta.types import ColumnsInputT
from metafora.meta.utils import parse_str_from_enum

from ..core import MetaFeaturesColumnsExtractor
from ..variants import *


class CorrelationExtractor(MetaFeaturesColumnsExtractor):
    def __init__(
            self,
            numeric_columns: ColumnsInputT,
            correlations: List[TCorrelation],
            post_aggregators: List[TNumeric]
    ):
        super(CorrelationExtractor, self).__init__(numeric_columns)

        self.correlations = parse_str_from_enum(correlations)
        self.post_aggregators = parse_str_from_enum(post_aggregators)

    def _fit(
            self,
            df: pd.DataFrame,
            *args,
            **kwargs
    ):
        if self.is_empty:
            self.features = [
                (f"{corr}_{agg}", np.nan)
                for corr in self.correlations
                for agg in self.post_aggregators
            ]
            return

        n = len(self.columns)

        for corr in self.correlations:
            corr_df = df[self.columns].corr(method=corr)

            if n == 1:
                corr_series = pd.Series([0])

            else:
                corr_series = pd.Series([
                    corr_df.iloc[i, j] for i in range(n - 1) for j in range(i + 1, n)
                ])

            meta = corr_series.apply(self.post_aggregators)

            self.features += list(
                zip(
                    map(lambda agg: f"{corr}_{agg}", self.post_aggregators),
                    meta.values
                )
            )

    def __repr__(self):
        stages = [
            f"{correlation} => {post_agg}"
            for correlation in self.correlations
            for post_agg in self.post_aggregators
        ]

        stages = "\n\t\t".join(stages)

        return f"{self.__class__.__name__}(\n\t\t{stages}\n\t)"


__all__ = [
    "CorrelationExtractor"
]
