from math import sqrt
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _safe_curve_score(metric_fn, y_true: np.ndarray, prob: np.ndarray) -> Optional[float]:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(metric_fn(y_true, prob))


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= threshold).astype(int)
    matrix = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    acsa = None if sensitivity is None or specificity is None else float((sensitivity + specificity) / 2.0)
    gm = None if sensitivity is None or specificity is None else float(sqrt(max(sensitivity * specificity, 0.0)))
    f1_value = float(f1_score(y_true, pred, zero_division=0))
    roc_auc = _safe_curve_score(roc_auc_score, y_true, prob)
    pr_auc = _safe_curve_score(average_precision_score, y_true, prob)

    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "acsa": acsa,
        "gm": gm,
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1_score": f1_value,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            pred,
            labels=[0, 1],
            target_names=["benign", "malware"],
            output_dict=True,
            zero_division=0,
        ),
    }
    metrics["auc"] = roc_auc
    metrics["f1"] = f1_value
    return metrics


def metric_for_best_model(metrics: Dict[str, object], primary: str = "pr_auc") -> float:
    for key in [primary, "f1_score", "recall", "accuracy"]:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return -1.0
