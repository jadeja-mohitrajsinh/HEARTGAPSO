"""Evaluate the saved HeartGAPSO model on labeled data.

By default this recreates the stratified holdout used by ``train.py``.  Supply
``--data`` with a labeled CSV to score an independent test set instead.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data_loader import FEATURE_NAMES, load_cleveland


DEFAULT_DATA = Path("heart+disease/processed.cleveland.data")


def load_labeled_csv(path: Path, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a headered external CSV and convert UCI targets 1--4 to positive."""
    frame = pd.read_csv(path, na_values=["?"])
    missing_features = set(FEATURE_NAMES) - set(frame.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {sorted(missing_features)}")
    if target_column not in frame.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(frame.columns)}"
        )
    target = pd.to_numeric(frame[target_column], errors="coerce")
    if target.isna().any():
        raise ValueError("The target column contains missing or non-numeric values")
    if not set(target.astype(int).unique()).issubset({0, 1, 2, 3, 4}):
        raise ValueError("Targets must be binary (0/1) or UCI values from 0 through 4")
    features = frame[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    return features, (target > 0).astype(int)


def calculate_metrics(y_true, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "threshold": threshold,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the saved HeartGAPSO model")
    parser.add_argument(
        "--data",
        type=Path,
        help="Labeled headered CSV for external testing. Omit to reproduce the original holdout.",
    )
    parser.add_argument("--target-column", default="target", help="Label column in --data (default: target)")
    parser.add_argument(
        "--full-cleveland",
        action="store_true",
        help="Score all Cleveland rows when --data is omitted (not an independent estimate).",
    )
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON report")
    args = parser.parse_args()

    config = json.loads(Path("models/model_config.json").read_text(encoding="utf-8"))
    with Path("models/gapso_random_forest.pkl").open("rb") as file:
        model = pickle.load(file)

    if args.data:
        X, y = load_labeled_csv(args.data, args.target_column)
        evaluation_set = f"external CSV: {args.data}"
    else:
        X, y = load_cleveland(DEFAULT_DATA)
        if args.full_cleveland:
            evaluation_set = "all Cleveland rows (includes model-training rows)"
        else:
            _, X, _, y = train_test_split(
                X, y, test_size=0.20, stratify=y, random_state=int(config["seed"]),
            )
            evaluation_set = "recreated original untouched Cleveland holdout"

    threshold = float(config.get("classification_threshold", getattr(model, "threshold", 0.5)))
    probabilities = model.predict_proba(X[FEATURE_NAMES])[:, 1]
    report = {
        "evaluation_set": evaluation_set,
        "selected_features": config["selected_features"],
        **calculate_metrics(y, probabilities, threshold),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
