from __future__ import annotations

import math

import numpy as np

from odyssey.evaluation.metrics import evaluate_model, expected_calibration_error


def test_metrics_for_easy_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate_model(
        y_true,
        probs,
        threshold=0.5,
        fixed_fpr_target=0.05,
        ece_bins=5,
        latency_ms_per_sample=0.3,
        parameter_count_value=10,
        training_time_s=1.2,
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert math.isclose(metrics["roc_auc"], 1.0)
    assert math.isclose(metrics["pr_auc"], 1.0)
    assert 0.0 <= expected_calibration_error(y_true, probs, n_bins=5) <= 1.0


