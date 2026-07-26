"""Получение вероятности дефолта из сохранённой модели без обучения."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from src.dataset import assemble_modeling_data, read_application_csv
from src.model_bundle import (
    CreditScoringModel,
    load_model_bundle,
    resolve_model_path,
)
from src.validation import validate_inference_data


LOGGER = logging.getLogger(__name__)


def prepare_inference_matrix(
    application: pd.DataFrame,
    bundle: CreditScoringModel,
    *,
    feature_tables: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Собрать wide-матрицу только когда input ещё не содержит агрегаты."""
    if set(bundle.feature_names).issubset(application.columns):
        return application
    return assemble_modeling_data(
        application,
        feature_set=bundle.feature_set,
        feature_tables=feature_tables,
    )


def predict_dataframe(
    application: pd.DataFrame,
    bundle: CreditScoringModel,
    *,
    threshold: float | None = None,
    feature_tables: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Вернуть ID, probability, class и risk category для каждой строки."""
    validate_inference_data(application)
    modeling_data = prepare_inference_matrix(
        application,
        bundle,
        feature_tables=feature_tables,
    )
    probabilities = bundle.predict_proba(modeling_data)
    actual_threshold = bundle.threshold if threshold is None else threshold
    predictions = (probabilities >= actual_threshold).astype(int)
    return pd.DataFrame(
        {
            "SK_ID_CURR": application["SK_ID_CURR"].to_numpy(),
            "default_probability": probabilities,
            "predicted_class": predictions,
            "risk_category": bundle.assign_risk(probabilities),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    """Создать parser команды инференса."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--threshold", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Загрузить CSV и bundle, записать predictions CSV."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    arguments = build_parser().parse_args(argv)
    model_path = resolve_model_path(arguments.model)
    bundle = load_model_bundle(model_path)
    application = read_application_csv(
        arguments.input,
        require_target=False,
    )
    predictions = predict_dataframe(
        application,
        bundle,
        threshold=arguments.threshold,
    )
    output_path = arguments.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    LOGGER.info("Модель: %s", model_path)
    LOGGER.info("Предсказания сохранены: %s (%s строк)", output_path, len(predictions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
