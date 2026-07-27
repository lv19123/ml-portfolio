"""Метрики бинарного кредитного скоринга и выбор аналитического порога."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _validated_arrays(
    target: Sequence[int],
    probabilities: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Проверить target и score перед расчётом метрик."""
    y_true = np.asarray(target, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or scores.ndim != 1:
        raise ValueError("target и probabilities должны быть одномерными")
    if len(y_true) == 0 or len(y_true) != len(scores):
        raise ValueError("target и probabilities должны иметь равную длину")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("target должен содержать только 0 и 1")
    if np.unique(y_true).size != 2:
        raise ValueError("Для оценки должны присутствовать оба класса")
    if not np.isfinite(scores).all():
        raise ValueError("probabilities содержит NaN или infinity")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities должны находиться в диапазоне [0, 1]")
    return y_true, scores


def ks_statistic(
    target: Sequence[int],
    probabilities: Sequence[float],
) -> float:
    """Рассчитать максимальный разрыв TPR и FPR (KS statistic)."""
    y_true, scores = _validated_arrays(target, probabilities)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
    return float(np.max(true_positive_rate - false_positive_rate))


def select_f1_threshold(
    target: Sequence[int],
    probabilities: Sequence[float],
) -> float:
    """Выбрать порог максимального F1 только на validation/OOF данных."""
    y_true, scores = _validated_arrays(target, probabilities)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = int(np.nanargmax(f1_values))
    return float(np.clip(thresholds[best], 1e-6, 1 - 1e-6))


def calculate_binary_metrics(
    target: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Рассчитать ranking, probability и threshold-dependent метрики."""
    if not 0 < threshold < 1:
        raise ValueError("threshold должен находиться строго между 0 и 1")
    y_true, scores = _validated_arrays(target, probabilities)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "n_rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "ks_statistic": ks_statistic(y_true, scores),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def summarize_fold_metrics(
    fold_metrics: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Свести ROC-AUC и Average Precision folds к mean/std."""
    if not fold_metrics:
        raise ValueError("Список fold metrics пуст")
    summary: dict[str, float] = {}
    for metric in ("roc_auc", "average_precision"):
        values = np.asarray(
            [float(result[metric]) for result in fold_metrics],
            dtype=float,
        )
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std())
    return summary
