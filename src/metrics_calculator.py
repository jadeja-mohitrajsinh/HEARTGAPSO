import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred, y_prob=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics['roc_auc'] = 0.5
        try:
            metrics['pr_auc'] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics['pr_auc'] = 0.5
    else:
        metrics['roc_auc'] = 0.5
        metrics['pr_auc'] = 0.5
    return metrics


def format_metrics(metrics: dict) -> dict:
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.4f}"
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
            formatted[key] = f"{np.mean(value):.4f} ± {np.std(value):.4f}"
        else:
            formatted[key] = str(value)
    return formatted
