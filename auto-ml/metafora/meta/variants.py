import enum
from typing import Union


class NumericaVariants(enum.Enum):
    Sum = "sum"
    Max = "max"
    Min = "min"
    Std = "std"
    Mean = "mean"
    Median = "median"
    Variance = "var"
    Kurtosis = "kurtosis"
    Skew = "skew"


class CorrelationVariants(enum.Enum):
    Pearson = "pearson"
    Kendall = "kendall"
    Spearman = "spearman"


class CategoricalVariants(enum.Enum):
    DistinctClasses = "distinct_classes"
    Entropy = "entropy"


TNumeric = Union[NumericaVariants, str]
TCorrelation = Union[CorrelationVariants, str]
TCategorical = Union[CategoricalVariants, str]

__all__ = [
    "NumericaVariants",
    "CorrelationVariants",
    "CategoricalVariants",
    "TNumeric",
    "TCorrelation",
    "TCategorical"
]
