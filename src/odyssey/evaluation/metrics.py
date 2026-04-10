"""Evaluation metrics for Odyssey experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as skm


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (probs >= start) & (probs < end if end < 1.0 else probs <= end)
        if not np.any(mask):
            continue
        confidence = probs[mask].mean()
        accuracy = y_true[mask].mean()
        ece += abs(confidence - accuracy) * (mask.sum() / len(y_true))
    return float(ece)


def recall_at_fixed_fpr(y_true: np.ndarray, probs: np.ndarray, target_fpr: float) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = skm.roc_curve(y_true, probs)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(tpr[valid[-1]])


def parameter_count(model: Any) -> int:
    if hasattr(model, "parameters"):
        return int(sum(parameter.numel() for parameter in model.parameters()))
    if hasattr(model, "coef_"):
        return int(model.coef_.size + model.intercept_.size)
    if hasattr(model, "estimators_"):
        return int(sum(tree.tree_.node_count for tree in model.estimators_))
    return 0


def evaluate_model(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    fixed_fpr_target: float = 0.05,
    ece_bins: int = 10,
    latency_ms_per_sample: float = 0.0,
    parameter_count_value: int = 0,
    training_time_s: float = 0.0,
) -> dict[str, float]:
    """Compute the main metric suite for binary intrusion detection."""

    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    result = {
        "accuracy": float(skm.accuracy_score(y_true, preds)),
        "precision": float(skm.precision_score(y_true, preds, zero_division=0)),
        "recall": float(skm.recall_score(y_true, preds, zero_division=0)),
        "f1": float(skm.f1_score(y_true, preds, zero_division=0)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, preds)),
        "brier_score": float(np.mean((probs - y_true) ** 2)),
        "ece": expected_calibration_error(y_true, probs, n_bins=ece_bins),
        "recall_at_fixed_fpr": recall_at_fixed_fpr(y_true, probs, fixed_fpr_target),
        "latency_ms_per_sample": float(latency_ms_per_sample),
        "parameter_count": float(parameter_count_value),
        "training_time_s": float(training_time_s),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    if len(np.unique(y_true)) < 2:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")
    else:
        result["roc_auc"] = float(skm.roc_auc_score(y_true, probs))
        result["pr_auc"] = float(skm.average_precision_score(y_true, probs))
    return result


