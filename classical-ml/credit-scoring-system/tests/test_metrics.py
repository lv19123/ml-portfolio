import numpy as np
import pytest

from src.metrics import (
    calculate_binary_metrics,
    ks_statistic,
    select_f1_threshold,
    summarize_fold_metrics,
)


def test_binary_metrics_include_credit_scoring_metrics():
    target = [0, 0, 0, 1, 1, 1]
    probabilities = [0.05, 0.20, 0.45, 0.40, 0.75, 0.95]

    metrics = calculate_binary_metrics(
        target,
        probabilities,
        threshold=0.5,
    )

    expected = {
        "roc_auc",
        "average_precision",
        "brier_score",
        "ks_statistic",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    }
    assert expected.issubset(metrics)
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["average_precision"] <= 1
    assert sum(
        metrics[name]
        for name in [
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ]
    ) == len(target)


def test_threshold_and_ks_are_valid():
    target = [0, 0, 1, 1]
    probabilities = [0.1, 0.3, 0.6, 0.9]

    assert 0 < select_f1_threshold(target, probabilities) < 1
    assert ks_statistic(target, probabilities) == pytest.approx(1.0)


def test_metrics_reject_invalid_probabilities():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calculate_binary_metrics(
            [0, 1],
            [-0.1, 1.1],
            threshold=0.5,
        )

    with pytest.raises(ValueError, match="оба класса"):
        calculate_binary_metrics(
            [0, 0],
            [0.1, 0.2],
            threshold=0.5,
        )


def test_fold_summary_uses_real_fold_values():
    summary = summarize_fold_metrics(
        [
            {"roc_auc": 0.7, "average_precision": 0.2},
            {"roc_auc": 0.9, "average_precision": 0.4},
        ]
    )

    assert summary["roc_auc_mean"] == pytest.approx(0.8)
    assert summary["average_precision_mean"] == pytest.approx(0.3)
    assert np.isfinite(list(summary.values())).all()
