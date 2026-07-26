"""Единый preprocessing для обучения и инференса."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE


def make_logistic_pipeline(
    *,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Создать baseline Pipeline без утечки preprocessing между folds."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_exclude=np.number),
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )


def infer_categorical_features(data: pd.DataFrame) -> list[str]:
    """Вернуть категориальные колонки в стабильном исходном порядке."""
    return data.select_dtypes(exclude=np.number).columns.tolist()


def prepare_catboost_features(
    data: pd.DataFrame,
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """Применить тот же минимальный preprocessing, что использует CatBoost."""
    prepared = data.copy()
    missing = [
        column
        for column in categorical_features
        if column not in prepared.columns
    ]
    if missing:
        raise ValueError(
            f"Нет категориальных признаков из metadata: {missing[:10]}"
        )
    columns = list(categorical_features)
    if columns:
        prepared[columns] = prepared[columns].fillna("Unknown").astype(str)
    return prepared
