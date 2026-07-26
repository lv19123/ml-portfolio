"""Единый реестр метрик моделирующих экспериментов."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import TABLES_DIR


MODEL_METRICS_PATH = TABLES_DIR / "model_metrics.csv"

MODEL_METRICS_COLUMNS = [
    "experiment",
    "notebook",
    "model",
    "feature_set",
    "source_tables",
    "device",
    "n_train",
    "n_holdout",
    "n_features",
    "cv_folds",
    "best_iteration",
    "cv_roc_auc",
    "cv_roc_auc_std",
    "cv_pr_auc",
    "cv_pr_auc_std",
    "holdout_roc_auc",
    "holdout_pr_auc",
]

EXPERIMENT_ORDER = [
    "application_logistic",
    "application_catboost",
    "application_bureau",
    "application_previous",
    "application_installments",
    "application_pos_cash",
    "application_credit_card",
    "application_all_features",
]


def _validate_columns(
    columns: list[str],
    *,
    source_name: str,
    require_order: bool = True,
) -> None:
    """Проверить точное соответствие единой схеме метрик."""
    if (
        set(columns) == set(MODEL_METRICS_COLUMNS)
        and (
            not require_order
            or columns == MODEL_METRICS_COLUMNS
        )
    ):
        return

    missing = [
        column
        for column in MODEL_METRICS_COLUMNS
        if column not in columns
    ]
    extra = [
        column
        for column in columns
        if column not in MODEL_METRICS_COLUMNS
    ]
    raise ValueError(
        f"Некорректная схема {source_name}. "
        f"Ожидаются столбцы в порядке {MODEL_METRICS_COLUMNS}. "
        f"Отсутствуют: {missing}. Лишние: {extra}."
    )


def _sort_results(results: pd.DataFrame) -> pd.DataFrame:
    """Отсортировать строки в порядке выполнения экспериментов."""
    order = {
        experiment: position
        for position, experiment in enumerate(EXPERIMENT_ORDER)
    }
    sorted_results = (
        results
        .assign(
            _experiment_order=results["experiment"].map(order).fillna(
                len(order)
            )
        )
        .sort_values(
            ["_experiment_order", "experiment", "model"],
            kind="stable",
        )
        .drop(columns="_experiment_order")
        .reset_index(drop=True)
    )
    return sorted_results[MODEL_METRICS_COLUMNS]


def _write_csv_atomically(
    results: pd.DataFrame,
    path: Path,
) -> None:
    """Записать CSV через временный файл в той же директории."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        results.to_csv(
            temporary_path,
            index=False,
        )
        os.replace(
            temporary_path,
            path,
        )
    finally:
        temporary_path.unlink(
            missing_ok=True,
        )


def save_experiment_result(
    result: Mapping[str, Any],
    path: Path | None = None,
) -> pd.DataFrame:
    """Добавить или заменить одну строку в едином CSV с метриками.

    Уникальность строки определяется парой ``experiment`` + ``model``.
    Функция возвращает полную таблицу после upsert.
    """
    results_path = Path(path) if path is not None else MODEL_METRICS_PATH
    result_columns = list(result.keys())
    _validate_columns(
        result_columns,
        source_name="строки эксперимента",
        require_order=False,
    )

    current_result = pd.DataFrame(
        [
            {
                column: result[column]
                for column in MODEL_METRICS_COLUMNS
            }
        ],
        columns=MODEL_METRICS_COLUMNS,
    )
    for holdout_column in [
        "holdout_roc_auc",
        "holdout_pr_auc",
    ]:
        if current_result.loc[0, holdout_column] is None:
            current_result.loc[0, holdout_column] = np.nan

    if results_path.exists():
        all_results = pd.read_csv(results_path)
        _validate_columns(
            all_results.columns.tolist(),
            source_name=str(results_path),
        )
        same_result = (
            all_results["experiment"].eq(
                current_result.loc[0, "experiment"]
            )
            & all_results["model"].eq(
                current_result.loc[0, "model"]
            )
        )
        all_results = all_results.loc[~same_result].copy()
        all_results = pd.concat(
            [all_results, current_result],
            ignore_index=True,
        )
    else:
        all_results = current_result.copy()

    all_results = _sort_results(all_results)
    _write_csv_atomically(
        all_results,
        results_path,
    )
    return all_results
