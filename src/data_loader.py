"""Canonical Cleveland data loading and validation."""

from pathlib import Path

import pandas as pd


FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
_COLUMNS = [*FEATURE_NAMES, "target"]


def load_cleveland(data_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load local UCI Cleveland data and convert severity 1--4 to disease."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cleveland data file was not found: {path}")
    frame = pd.read_csv(path, header=None, names=_COLUMNS, na_values=["?"], dtype=str)
    if frame.shape[1] != len(_COLUMNS):
        raise ValueError(f"Expected {len(_COLUMNS)} columns in {path}, found {frame.shape[1]}")
    features = frame[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    target = pd.to_numeric(frame["target"], errors="coerce")
    if target.isna().any():
        rows = target.index[target.isna()].tolist()[:5]
        raise ValueError(f"Target contains missing or non-numeric values at rows {rows}")
    if not target.isin([0, 1, 2, 3, 4]).all():
        values = sorted(target[~target.isin([0, 1, 2, 3, 4])].unique().tolist())
        raise ValueError(f"Unexpected Cleveland target values: {values}")
    if len(features) < 20 or target.nunique() < 2:
        raise ValueError("Dataset is too small or does not contain both target classes")
    return features, (target > 0).astype(int).rename("target")
