"""Версионированный bundle модели и её preprocessing-контракта."""

from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import prepare_catboost_features
from src.validation import validate_required_features
from src.config import DEFAULT_MODEL_BUNDLE_PATH, LEGACY_LOGISTIC_MODEL_PATH


def predict_positive_probability(
    estimator: Any,
    features: pd.DataFrame,
) -> np.ndarray:
    """Безопасно вызвать predict_proba и вернуть positive-class score.

    NumPy 2.2 на некоторых Accelerate/BLAS-сборках macOS может выдавать
    ложные ``encountered in matmul`` warnings для конечных матриц. Warning
    подавляется узко, а результат ниже обязательно проверяется.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        probabilities = np.asarray(
            estimator.predict_proba(features),
            dtype=float,
        )
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            "predict_proba должен вернуть матрицу размера (n_rows, 2)"
        )
    positive = probabilities[:, 1]
    if (
        len(positive) != len(features)
        or not np.isfinite(positive).all()
        or ((positive < 0) | (positive > 1)).any()
    ):
        raise ValueError("Модель вернула некорректные вероятности")
    return positive


@dataclass
class CreditScoringModel:
    """Модель вместе со схемой признаков, порогом и risk bands."""

    estimator: Any
    feature_names: tuple[str, ...]
    model_type: str
    feature_set: str = "application"
    categorical_features: tuple[str, ...] = ()
    threshold: float = 0.5
    risk_cutoffs: tuple[float, float] = (0.25, 0.50)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Проверить неизменяемый контракт bundle после создания/загрузки."""
        if self.model_type not in {"logistic", "catboost"}:
            raise ValueError(f"Неизвестный model_type: {self.model_type}")
        if not self.feature_names:
            raise ValueError("Bundle не содержит feature_names")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature_names содержит дубликаты")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold должен находиться строго между 0 и 1")
        low, high = self.risk_cutoffs
        if not 0 < low < high < 1:
            raise ValueError(
                "risk_cutoffs должны удовлетворять 0 < low < high < 1"
            )

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Выбрать и преобразовать признаки точно в обучающем порядке."""
        validate_required_features(data, self.feature_names)
        features = data.loc[:, list(self.feature_names)].copy()
        if self.model_type == "catboost":
            features = prepare_catboost_features(
                features,
                self.categorical_features,
            )
        return features

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Вернуть одномерный вектор вероятностей TARGET=1."""
        features = self.prepare_features(data)
        return predict_positive_probability(self.estimator, features)

    def predict(
        self,
        data: pd.DataFrame,
        *,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Вернуть бинарный класс по сохранённому или переданному порогу."""
        actual_threshold = self.threshold if threshold is None else threshold
        if not 0 < actual_threshold < 1:
            raise ValueError("threshold должен находиться строго между 0 и 1")
        return (self.predict_proba(data) >= actual_threshold).astype(int)

    def assign_risk(self, probabilities: np.ndarray) -> np.ndarray:
        """Сегментировать score на low/medium/high по сохранённым границам."""
        values = np.asarray(probabilities, dtype=float)
        low, high = self.risk_cutoffs
        return np.select(
            [values < low, values < high],
            ["low", "medium"],
            default="high",
        )


def save_model_bundle(
    bundle: CreditScoringModel,
    path: Path,
) -> Path:
    """Атомарно сохранить joblib bundle в указанном файле."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}-",
        suffix=".joblib.tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(bundle, temporary_path)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target


def load_model_bundle(path: Path) -> CreditScoringModel:
    """Загрузить новый bundle или совместимый legacy sklearn Pipeline."""
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Файл модели не найден: {model_path}. "
            "Сначала выполните python3 -m src.train."
        )
    artifact = joblib.load(model_path)
    if isinstance(artifact, CreditScoringModel):
        return artifact

    feature_names = tuple(
        str(column)
        for column in getattr(artifact, "feature_names_in_", ())
    )
    if hasattr(artifact, "predict_proba") and feature_names:
        return CreditScoringModel(
            estimator=artifact,
            feature_names=feature_names,
            model_type="logistic",
            feature_set="application",
            metadata={
                "legacy_artifact": True,
                "source_path": str(model_path),
            },
        )
    raise TypeError(
        f"Неподдерживаемый формат модели в {model_path}: "
        f"{type(artifact).__name__}"
    )


def resolve_model_path(path: Path | None = None) -> Path:
    """Выбрать явный, новый default или совместимый legacy-артефакт."""
    if path is not None:
        return Path(path).expanduser().resolve()
    if DEFAULT_MODEL_BUNDLE_PATH.is_file():
        return DEFAULT_MODEL_BUNDLE_PATH
    if LEGACY_LOGISTIC_MODEL_PATH.is_file():
        return LEGACY_LOGISTIC_MODEL_PATH
    return DEFAULT_MODEL_BUNDLE_PATH
