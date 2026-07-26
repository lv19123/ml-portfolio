"""Загрузка, split и сборка модельной матрицы Home Credit."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLIENT_SPLIT_PATH,
    INTERIM_DATA_DIR,
    RANDOM_STATE,
    find_data_file,
)
from src.features import (
    build_bureau_features,
    build_credit_card_features,
    build_installments_features,
    build_pos_cash_features,
    build_previous_application_features,
)
from src.validation import (
    merge_feature_table,
    validate_application_data,
    validate_feature_table,
)


LOGGER = logging.getLogger(__name__)

FEATURE_TABLE_FILENAMES = {
    "bureau": "bureau_features.csv",
    "previous": "previous_application_features.csv",
    "installments": "installments_features.csv",
    "pos_cash": "pos_cash_features.csv",
    "credit_card": "credit_card_features.csv",
}
FEATURE_SET_TABLES = {
    "application": (),
    "pos_cash": ("pos_cash",),
    "all": (
        "bureau",
        "previous",
        "installments",
        "pos_cash",
        "credit_card",
    ),
}


def read_application_csv(
    path: Path,
    *,
    require_target: bool,
) -> pd.DataFrame:
    """Прочитать application CSV и проверить train/inference-контракт."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Файл application не найден: {source}")
    data = pd.read_csv(source)
    validate_application_data(data, require_target=require_target)
    if "DAYS_EMPLOYED" in data.columns:
        data["DAYS_EMPLOYED"] = data["DAYS_EMPLOYED"].replace(365243, np.nan)
    return data


def _build_feature_table(table_name: str) -> pd.DataFrame:
    """Построить один агрегат из сырых CSV."""
    if table_name == "bureau":
        return build_bureau_features(
            pd.read_csv(find_data_file("bureau.csv")),
            pd.read_csv(find_data_file("bureau_balance.csv")),
        )
    if table_name == "previous":
        return build_previous_application_features(
            pd.read_csv(find_data_file("previous_application.csv"))
        )
    if table_name == "installments":
        return build_installments_features(
            pd.read_csv(find_data_file("installments_payments.csv"))
        )
    if table_name == "pos_cash":
        return build_pos_cash_features(
            pd.read_csv(find_data_file("POS_CASH_balance.csv"))
        )
    if table_name == "credit_card":
        return build_credit_card_features(
            pd.read_csv(find_data_file("credit_card_balance.csv"))
        )
    raise ValueError(f"Неизвестная таблица признаков: {table_name}")


def load_or_build_feature_table(
    table_name: str,
    *,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Прочитать сохранённый агрегат либо воспроизводимо построить его."""
    if table_name not in FEATURE_TABLE_FILENAMES:
        raise ValueError(f"Неизвестная таблица признаков: {table_name}")
    output_path = INTERIM_DATA_DIR / FEATURE_TABLE_FILENAMES[table_name]
    if output_path.is_file() and not rebuild:
        features = pd.read_csv(output_path)
        validate_feature_table(features)
        return features

    LOGGER.info("Строится таблица признаков %s", table_name)
    features = _build_feature_table(table_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    LOGGER.info("Таблица признаков сохранена: %s", output_path)
    return features


def assemble_modeling_data(
    application: pd.DataFrame,
    *,
    feature_set: str,
    feature_tables: Mapping[str, pd.DataFrame] | None = None,
    rebuild_features: bool = False,
) -> pd.DataFrame:
    """Добавить к application таблицы выбранного набора признаков."""
    if feature_set not in FEATURE_SET_TABLES:
        raise ValueError(
            f"feature_set должен быть одним из {sorted(FEATURE_SET_TABLES)}"
        )
    result = application.copy()
    supplied = feature_tables or {}
    for table_name in FEATURE_SET_TABLES[feature_set]:
        features = supplied.get(table_name)
        if features is None:
            features = load_or_build_feature_table(
                table_name,
                rebuild=rebuild_features,
            )
        result = merge_feature_table(result, features)
    return result


def create_or_load_client_split(
    application: pd.DataFrame,
    *,
    path: Path = CLIENT_SPLIT_PATH,
    holdout_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Создать или проверить единый стратифицированный client split."""
    validate_application_data(application)
    split_path = Path(path).expanduser().resolve()
    if split_path.is_file():
        split = pd.read_csv(split_path)
    else:
        _, holdout_ids = train_test_split(
            application["SK_ID_CURR"],
            test_size=holdout_size,
            stratify=application["TARGET"],
            random_state=random_state,
        )
        split = application[["SK_ID_CURR"]].copy()
        split["split"] = "train"
        split.loc[
            split["SK_ID_CURR"].isin(holdout_ids),
            "split",
        ] = "holdout"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(split_path, index=False)

    expected_columns = ["SK_ID_CURR", "split"]
    if split.columns.tolist() != expected_columns:
        raise ValueError(
            f"Некорректная схема client split: {split.columns.tolist()}"
        )
    if split["SK_ID_CURR"].isna().any() or not split["SK_ID_CURR"].is_unique:
        raise ValueError("SK_ID_CURR в client split должен быть уникальным")
    if set(split["split"]) != {"train", "holdout"}:
        raise ValueError("client split должен содержать train и holdout")
    if set(split["SK_ID_CURR"]) != set(application["SK_ID_CURR"]):
        raise ValueError("client split не соответствует application_train")
    return split


def split_modeling_data(
    modeling_data: pd.DataFrame,
    client_split: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разделить модельную таблицу по сохранённым идентификаторам."""
    with_split = modeling_data.merge(
        client_split,
        on="SK_ID_CURR",
        how="inner",
        validate="one_to_one",
    )
    if len(with_split) != len(modeling_data):
        raise ValueError("При добавлении client split потеряны строки")
    train = with_split.loc[with_split["split"].eq("train")].copy()
    holdout = with_split.loc[with_split["split"].eq("holdout")].copy()
    if train.empty or holdout.empty:
        raise ValueError("Train или holdout оказался пустым")
    return train.reset_index(drop=True), holdout.reset_index(drop=True)


def get_model_feature_names(data: pd.DataFrame) -> list[str]:
    """Вернуть признаки без target, идентификатора и технического split."""
    excluded = {"TARGET", "SK_ID_CURR", "split"}
    features = [column for column in data.columns if column not in excluded]
    if not features:
        raise ValueError("После исключения служебных колонок нет признаков")
    return features


def stratified_sample(
    data: pd.DataFrame,
    *,
    size: int | None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Получить воспроизводимую стратифицированную подвыборку."""
    if size is None or size >= len(data):
        return data.reset_index(drop=True)
    if size < 2:
        raise ValueError("Размер sample должен быть не меньше 2")
    sample, _ = train_test_split(
        data,
        train_size=size,
        stratify=data["TARGET"],
        random_state=random_state,
    )
    return sample.reset_index(drop=True)


def required_feature_tables(feature_set: str) -> Sequence[str]:
    """Вернуть имена агрегатов для заданного feature set."""
    if feature_set not in FEATURE_SET_TABLES:
        raise ValueError(f"Неизвестный feature_set: {feature_set}")
    return FEATURE_SET_TABLES[feature_set]
