"""Random Forest construction and the serializable inference wrapper."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def make_random_forest(params: dict, seed: int) -> RandomForestClassifier:
    """Build a deterministic single-process RF for repeatable local runs."""
    return RandomForestClassifier(random_state=seed, n_jobs=1, **params)


class InferencePipeline:
    """Keep preprocessing, selected columns, classifier, and threshold together."""

    def __init__(self, preprocessing, classifier: RandomForestClassifier, indices: Sequence[int], threshold: float,
                 feature_names: Sequence[str] | None = None) -> None:
        self.preprocessing = preprocessing
        self.classifier = classifier
        self.indices = list(indices)
        self.threshold = float(threshold)
        self.feature_names = list(feature_names) if feature_names is not None else None

    def _select(self, raw: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(raw, pd.DataFrame):
            raise TypeError("InferencePipeline expects a pandas DataFrame with named input columns")
        feature_names = getattr(self, "feature_names", None)
        if feature_names is not None:
            missing = [name for name in feature_names if name not in raw.columns]
            if missing:
                raise ValueError(f"Missing model feature columns: {missing}")
            return raw.loc[:, feature_names]
        return raw.iloc[:, self.indices]  # compatibility with pre-fix pickles

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(self.preprocessing.transform(self._select(raw)))

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(raw)[:, 1] >= self.threshold).astype(int)
