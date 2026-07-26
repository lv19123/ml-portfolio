"""Обучение baseline или основной credit-scoring модели из CLI."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

from src.config import (
    CLIENT_SPLIT_PATH,
    DEFAULT_MODEL_BUNDLE_PATH,
    DEFAULT_MODEL_METADATA_PATH,
    RANDOM_STATE,
    find_data_file,
)
from src.dataset import (
    assemble_modeling_data,
    create_or_load_client_split,
    get_model_feature_names,
    read_application_csv,
    split_modeling_data,
    stratified_sample,
)
from src.metrics import (
    calculate_binary_metrics,
    select_f1_threshold,
    summarize_fold_metrics,
)
from src.model_bundle import (
    CreditScoringModel,
    predict_positive_probability,
    save_model_bundle,
)
from src.model_config import get_catboost_gpu_count
from src.preprocessing import (
    infer_categorical_features,
    make_logistic_pipeline,
    prepare_catboost_features,
)
from src.utils import runtime_versions, write_json_atomically


LOGGER = logging.getLogger(__name__)


def _catboost_device_config(device: str) -> dict[str, Any]:
    """Преобразовать CLI device в параметры CatBoost с явной проверкой."""
    gpu_count = get_catboost_gpu_count()
    if device == "gpu":
        if gpu_count < 1:
            raise RuntimeError("Запрошен GPU, но CatBoost не обнаружил GPU")
        return {"task_type": "GPU", "devices": "0"}
    if device == "auto" and gpu_count > 0:
        return {"task_type": "GPU", "devices": "0"}
    return {"task_type": "CPU", "thread_count": -1}


def _make_estimator(
    *,
    model_type: str,
    random_state: int,
    iterations: int,
    learning_rate: float,
    depth: int,
    device: str,
) -> Any:
    """Создать новый estimator для одного fold или финального fit."""
    if model_type == "logistic":
        return make_logistic_pipeline(random_state=random_state)
    if model_type != "catboost":
        raise ValueError(f"Неизвестный model_type: {model_type}")
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=random_state,
        allow_writing_files=False,
        verbose=False,
        **_catboost_device_config(device),
    )


def _fit_estimator(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    model_type: str,
    categorical_features: list[str],
) -> None:
    """Обучить estimator с правильным интерфейсом категорий."""
    if model_type == "catboost":
        estimator.fit(
            features,
            target,
            cat_features=categorical_features,
        )
    else:
        estimator.fit(features, target)


def _risk_cutoffs(probabilities: np.ndarray) -> tuple[float, float]:
    """Получить описательные 50%/80% границы score из OOF."""
    low, high = np.quantile(probabilities, [0.50, 0.80]).tolist()
    low = float(np.clip(low, 1e-4, 1 - 2e-4))
    high = float(np.clip(high, low + 1e-4, 1 - 1e-4))
    return low, high


def train_model(
    modeling_data: pd.DataFrame,
    client_split: pd.DataFrame,
    *,
    model_type: str,
    feature_set: str,
    cv_folds: int = 3,
    iterations: int = 996,
    learning_rate: float = 0.05,
    depth: int = 6,
    device: str = "auto",
    threshold_strategy: str = "f1",
    train_sample_size: int | None = None,
    holdout_sample_size: int | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[CreditScoringModel, dict[str, Any]]:
    """Выполнить OOF-validation, final fit и однократную holdout-оценку."""
    if cv_folds < 2:
        raise ValueError("cv_folds должен быть не меньше 2")
    train_data, holdout_data = split_modeling_data(
        modeling_data,
        client_split,
    )
    full_train_size = len(train_data)
    full_holdout_size = len(holdout_data)
    train_data = stratified_sample(
        train_data,
        size=train_sample_size,
        random_state=random_state,
    )
    holdout_data = stratified_sample(
        holdout_data,
        size=holdout_sample_size,
        random_state=random_state,
    )
    feature_names = get_model_feature_names(modeling_data)
    X_train = train_data[feature_names]
    y_train = train_data["TARGET"].astype(int)
    X_holdout = holdout_data[feature_names]
    y_holdout = holdout_data["TARGET"].astype(int)

    categorical_features: list[str] = []
    if model_type == "catboost":
        categorical_features = infer_categorical_features(X_train)
        X_train = prepare_catboost_features(
            X_train,
            categorical_features,
        )
        X_holdout = prepare_catboost_features(
            X_holdout,
            categorical_features,
        )

    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    oof_probabilities = np.zeros(len(X_train), dtype=float)
    fold_metrics: list[dict[str, Any]] = []
    for fold, (fit_index, validation_index) in enumerate(
        splitter.split(X_train, y_train),
        start=1,
    ):
        LOGGER.info("Обучение fold %s/%s", fold, cv_folds)
        estimator = _make_estimator(
            model_type=model_type,
            random_state=random_state,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            device=device,
        )
        _fit_estimator(
            estimator,
            X_train.iloc[fit_index],
            y_train.iloc[fit_index],
            model_type=model_type,
            categorical_features=categorical_features,
        )
        fold_probabilities = predict_positive_probability(
            estimator,
            X_train.iloc[validation_index],
        )
        oof_probabilities[validation_index] = fold_probabilities
        fold_result = calculate_binary_metrics(
            y_train.iloc[validation_index],
            fold_probabilities,
            threshold=0.5,
        )
        fold_result["fold"] = fold
        fold_metrics.append(fold_result)

    if threshold_strategy == "f1":
        threshold = select_f1_threshold(y_train, oof_probabilities)
    elif threshold_strategy == "fixed":
        threshold = 0.5
    else:
        raise ValueError("threshold_strategy должен быть f1 или fixed")

    final_estimator = _make_estimator(
        model_type=model_type,
        random_state=random_state,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        device=device,
    )
    _fit_estimator(
        final_estimator,
        X_train,
        y_train,
        model_type=model_type,
        categorical_features=categorical_features,
    )
    holdout_probabilities = predict_positive_probability(
        final_estimator,
        X_holdout,
    )
    holdout_metrics = calculate_binary_metrics(
        y_holdout,
        holdout_probabilities,
        threshold=threshold,
    )
    cv_summary = summarize_fold_metrics(fold_metrics)
    is_smoke = (
        len(train_data) < full_train_size
        or len(holdout_data) < full_holdout_size
    )
    created_at = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": created_at,
        "is_smoke": is_smoke,
        "model_type": model_type,
        "feature_set": feature_set,
        "random_state": random_state,
        "cv_folds": cv_folds,
        "n_train": len(train_data),
        "n_holdout": len(holdout_data),
        "n_features": len(feature_names),
        "threshold_strategy": threshold_strategy,
        "threshold": threshold,
        "risk_cutoffs": list(_risk_cutoffs(oof_probabilities)),
        "cv": cv_summary,
        "fold_metrics": fold_metrics,
        "holdout": holdout_metrics,
        "parameters": {
            "iterations": iterations if model_type == "catboost" else None,
            "learning_rate": (
                learning_rate if model_type == "catboost" else None
            ),
            "depth": depth if model_type == "catboost" else None,
            "device": device,
        },
        "versions": runtime_versions(),
    }
    bundle = CreditScoringModel(
        estimator=final_estimator,
        feature_names=tuple(feature_names),
        model_type=model_type,
        feature_set=feature_set,
        categorical_features=tuple(categorical_features),
        threshold=threshold,
        risk_cutoffs=tuple(report["risk_cutoffs"]),
        metadata={
            "created_at": created_at,
            "is_smoke": is_smoke,
            "versions": report["versions"],
        },
    )
    return bundle, report


def build_parser() -> argparse.ArgumentParser:
    """Создать parser команды обучения."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="application_train.csv; по умолчанию ищется в data/raw",
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "catboost"],
        default="catboost",
    )
    parser.add_argument(
        "--features",
        choices=["application", "pos_cash", "all"],
        default="pos_cash",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=996)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=["f1", "fixed"],
        default="f1",
    )
    parser.add_argument("--train-sample-size", type=int)
    parser.add_argument("--holdout-sample-size", type=int)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_BUNDLE_PATH,
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_MODEL_METADATA_PATH,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Выполнить CLI обучения и сохранить model bundle с metadata."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    arguments = build_parser().parse_args(argv)
    if (
        arguments.train_sample_size is not None
        or arguments.holdout_sample_size is not None
    ) and Path(arguments.output).resolve() == DEFAULT_MODEL_BUNDLE_PATH.resolve():
        raise ValueError(
            "Smoke/subsample нельзя сохранять поверх default-модели. "
            "Передайте --output и --metadata-output."
        )
    input_path = (
        arguments.input
        if arguments.input is not None
        else find_data_file("application_train.csv")
    )
    application = read_application_csv(input_path, require_target=True)
    modeling_data = assemble_modeling_data(
        application,
        feature_set=arguments.features,
        rebuild_features=arguments.rebuild_features,
    )
    split = create_or_load_client_split(
        application,
        path=CLIENT_SPLIT_PATH,
    )
    bundle, report = train_model(
        modeling_data,
        split,
        model_type=arguments.model,
        feature_set=arguments.features,
        cv_folds=arguments.cv_folds,
        iterations=arguments.iterations,
        learning_rate=arguments.learning_rate,
        depth=arguments.depth,
        device=arguments.device,
        threshold_strategy=arguments.threshold_strategy,
        train_sample_size=arguments.train_sample_size,
        holdout_sample_size=arguments.holdout_sample_size,
    )
    model_path = save_model_bundle(bundle, arguments.output)
    metadata_path = write_json_atomically(report, arguments.metadata_output)
    LOGGER.info("Модель сохранена: %s", model_path)
    LOGGER.info("Metadata сохранена: %s", metadata_path)
    LOGGER.info(
        "Holdout ROC-AUC=%.6f, AP=%.6f",
        report["holdout"]["roc_auc"],
        report["holdout"]["average_precision"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
