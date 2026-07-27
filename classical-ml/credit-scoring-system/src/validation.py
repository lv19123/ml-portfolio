"""Проверки схем данных и безопасное объединение таблиц признаков."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _validate_frame_basics(data: pd.DataFrame, *, source_name: str) -> None:
    """Проверить общие свойства входной таблицы."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{source_name} должен быть pandas DataFrame")
    if data.empty:
        raise ValueError(f"{source_name} не содержит строк")
    if data.columns.has_duplicates:
        duplicated = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(
            f"{source_name} содержит повторяющиеся колонки: {duplicated}"
        )


def validate_application_data(
    application: pd.DataFrame,
    *,
    require_target: bool = True,
) -> None:
    """Проверить ключ и, для train-режима, бинарную целевую переменную."""
    _validate_frame_basics(application, source_name="application")

    required_columns = {"SK_ID_CURR"}
    if require_target:
        required_columns.add("TARGET")
    missing_columns = required_columns.difference(application.columns)
    if missing_columns:
        raise ValueError(f"Нет обязательных колонок: {sorted(missing_columns)}")
    if application["SK_ID_CURR"].isna().any():
        raise ValueError("SK_ID_CURR содержит пропуски")
    if not application["SK_ID_CURR"].is_unique:
        raise ValueError("SK_ID_CURR должен быть уникальным в application")
    if not require_target:
        return
    if application["TARGET"].isna().any():
        raise ValueError("TARGET содержит пропуски")
    if not application["TARGET"].isin([0, 1]).all():
        raise ValueError("TARGET должен содержать только 0 и 1")


def validate_inference_data(application: pd.DataFrame) -> None:
    """Проверить application-таблицу для инференса без целевой переменной."""
    validate_application_data(application, require_target=False)
    if "TARGET" in application.columns:
        raise ValueError(
            "TARGET не должен присутствовать во входных данных инференса"
        )


def validate_required_features(
    data: pd.DataFrame,
    required_features: Sequence[str],
) -> None:
    """Проверить наличие всех признаков, ожидаемых моделью."""
    required = list(required_features)
    if not required:
        raise ValueError("Список признаков модели пуст")
    if len(required) != len(set(required)):
        raise ValueError("Список признаков модели содержит дубликаты")
    missing = [column for column in required if column not in data.columns]
    if missing:
        preview = missing[:10]
        suffix = "" if len(missing) <= 10 else f" и ещё {len(missing) - 10}"
        raise ValueError(
            f"Во входных данных нет признаков модели: {preview}{suffix}"
        )


def validate_feature_table(
    features: pd.DataFrame,
    allowed_prefixes: Sequence[str] | None = None,
) -> None:
    """Проверить таблицу с одной строкой на клиента перед merge."""
    _validate_frame_basics(features, source_name="Таблица признаков")
    if "SK_ID_CURR" not in features.columns:
        raise ValueError("В таблице признаков нет SK_ID_CURR")
    if features["SK_ID_CURR"].isna().any():
        raise ValueError("SK_ID_CURR содержит пропуски в таблице признаков")
    if not features["SK_ID_CURR"].is_unique:
        raise ValueError("После агрегации SK_ID_CURR должен быть уникальным")

    forbidden_columns = {"TARGET", "split"}
    found_forbidden = forbidden_columns.intersection(features.columns)
    if found_forbidden:
        raise ValueError(
            f"В таблице признаков есть служебные колонки: {sorted(found_forbidden)}"
        )

    feature_columns = [column for column in features if column != "SK_ID_CURR"]
    if allowed_prefixes:
        wrong_names = [
            column
            for column in feature_columns
            if not column.startswith(tuple(allowed_prefixes))
        ]
        if wrong_names:
            raise ValueError(f"Неожиданные имена признаков: {wrong_names[:5]}")

    numeric = features[feature_columns].select_dtypes(include=np.number)
    if not numeric.empty and np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Таблица признаков содержит бесконечные значения")


def merge_feature_table(
    base_data: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Добавить агрегаты без размножения строк основной таблицы."""
    validate_feature_table(features)
    rows_before = len(base_data)
    result = base_data.merge(
        features,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    if len(result) != rows_before:
        raise ValueError("После merge изменилось количество строк")
    if not result["SK_ID_CURR"].is_unique:
        raise ValueError("После merge появились повторяющиеся клиенты")
    return result
