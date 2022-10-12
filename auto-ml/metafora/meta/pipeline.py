import functools
from typing import Union

import pandas as pd

from .core import MetaFeatures, MetaFeaturesExtractor


class Pipeline(MetaFeaturesExtractor):
    def __init__(self, *args: MetaFeaturesExtractor):
        super(Pipeline, self).__init__()
        self.extractors = args
        self.transformation = MetaFeatures()

    def _fit(
            self,
            df: pd.DataFrame,
            *args,
            **kwargs
    ):
        transformations = [
            e.fit_transform(df) for e in self.extractors
        ]

        self.transformation = functools.reduce(
            lambda acc, transform: acc.concat(transform),
            transformations,
            self.transformation
        )

    def _transform(
            self,
            df: pd.DataFrame,
            return_df: bool = False,
            inplace: bool = False
    ) -> Union[pd.DataFrame, MetaFeatures]:
        if not return_df:
            return self.transformation

        return self.transformation.apply(df, inplace)

    def concat(self, other: "Pipeline") -> "Pipeline":
        return Pipeline(*self.extractors, *other.extractors)

    def __repr__(self):
        return self.depth_repr(depth=1)

    def depth_repr(self, depth: int) -> str:
        nl, tab = "\n", "\t"

        repr_string = ""
        for e in self.extractors:
            repr_string += nl + tab * depth
            if isinstance(e, Pipeline):
                repr_string += e.depth_repr(depth + 1)
            else:
                repr_string += repr(e)

        return f"{self.__class__.__name__}({repr_string}{nl}{tab * (depth - 1)})"


__all__ = [
    "Pipeline"
]
