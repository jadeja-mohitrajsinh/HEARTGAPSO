"""Holdout evaluation and run diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)


def evaluate_model(pipeline, X, y, selected_indices, output_dir, threshold: float) -> dict:
    """Evaluate a completed pipeline once on the caller-provided holdout set."""
    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1_score": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "threshold": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "holdout_samples": int(len(y)),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(destination / "roc_curve.png", dpi=160)
    plt.close()
    display = ConfusionMatrixDisplay(np.array(metrics["confusion_matrix"]), display_labels=["Healthy", "Disease"])
    display.plot(cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(destination / "confusion_matrix.png", dpi=160)
    plt.close()
    return metrics


def save_optimization_diagnostics(history: list[dict], opt_result: dict, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if history:
        generations = [item["generation"] for item in history]
        best = [item["best_fitness"] for item in history]
        mean = [item["mean_fitness"] for item in history]
        counts = [float(np.mean(item["selected_feature_counts"])) for item in history]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(generations, best, label="best")
        axes[0].plot(generations, mean, label="mean")
        axes[0].set(xlabel="generation", ylabel="CV fitness")
        axes[0].legend()
        axes[1].plot(generations, counts)
        axes[1].set(xlabel="generation", ylabel="mean selected features")
        figure.tight_layout()
        figure.savefig(destination / "optimization_history.png", dpi=160)
        figure.savefig(destination / "feature_selection_history.png", dpi=160)
        plt.close(figure)
    summary = {"best_fitness": float(opt_result["fitness"]),
               "selected_features_count": int(sum(opt_result["features"])),
               "optimizer_evaluations": int(opt_result["evaluations"])}
    (destination / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
