from typing import List

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from metafora.meta.utils import parse_str_from_enum

from ..core import MetaFeaturesExtractor
from ..variants import TNumeric


class StructuredExtractor(MetaFeaturesExtractor):
    def __init__(self, label: str, aggregators: List[TNumeric], model):
        super().__init__()
        self.label = label
        self.aggregators = parse_str_from_enum(aggregators)
        self.model = model

    def _fit(self, df: pd.DataFrame, *args, **kwargs):
        X = df.drop(columns=[self.label]).fillna(-1)
        X = pd.get_dummies(X)
        y = LabelEncoder().fit_transform(df[self.label])

        self.model.fit(X, y)


class DecisionTreeClassifierExtractor(StructuredExtractor):
    def __init__(self, label: str, aggregators: List[TNumeric]):
        super().__init__(label, aggregators, model=DecisionTreeClassifier(max_depth=10))

    def _fit(self, df: pd.DataFrame, *args, **kwargs):
        super(DecisionTreeClassifierExtractor, self)._fit(df, *args, **kwargs)

        tree = self.model.tree_

        threshold = pd.Series(tree.threshold).apply(self.aggregators)
        child_left = pd.Series(tree.children_left).apply(self.aggregators)
        child_right = pd.Series(tree.children_right).apply(self.aggregators)

        n_leaves = ("tree_n_leaves", tree.n_leaves)
        node_count = ("tree_node_count", tree.node_count)

        threshold = [(f"tree_threshold_{stat}", value) for stat, value in zip(self.aggregators, threshold)]
        child_left = [(f"tree_child_left_{stat}", value) for stat, value in zip(self.aggregators, child_left)]
        child_right = [(f"tree_child_right_{stat}", value) for stat, value in zip(self.aggregators, child_right)]

        self.features = [n_leaves, node_count]
        self.features += threshold
        self.features += child_left
        self.features += child_right


class SVMExtractor(StructuredExtractor):
    def __init__(self, label: str, aggregators: List[TNumeric]):
        super().__init__(label, aggregators, model=LinearSVC(max_iter=5000, tol=1e-3, C=2, verbose=False))

    def _fit(self, df: pd.DataFrame, *args, **kwargs):
        super(SVMExtractor, self)._fit(df, *args, **kwargs)

        classes, features = self.model.coef_.shape

        df = pd.DataFrame(
            self.model.coef_.T,
            index=[f"coef_{i}" for i in range(features)],
            columns=[f"class_{i}" for i in range(classes)]
        )

        aggregations = df.apply(self.aggregators).apply(self.aggregators, axis=1)

        names = [f"svc_coef_{col}_{post}" for col in self.aggregators for post in self.aggregators]
        values = aggregations.values.flatten()

        self.features = list(zip(names, values))
