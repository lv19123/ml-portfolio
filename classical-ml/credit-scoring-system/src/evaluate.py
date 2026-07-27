"""Оценка сохранённой модели на фиксированном holdout и отчёты."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    CLIENT_SPLIT_PATH,
    FIGURES_DIR,
    TABLES_DIR,
    find_data_file,
)
from src.dataset import (
    assemble_modeling_data,
    read_application_csv,
    split_modeling_data,
)
from src.metrics import calculate_binary_metrics
from src.model_bundle import (
    CreditScoringModel,
    load_model_bundle,
    resolve_model_path,
)
from src.utils import write_json_atomically


LOGGER = logging.getLogger(__name__)


def evaluate_bundle(
    bundle: CreditScoringModel,
    modeling_data: pd.DataFrame,
    client_split: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Оценить bundle только на holdout и вернуть построчные результаты."""
    _, holdout = split_modeling_data(modeling_data, client_split)
    probabilities = bundle.predict_proba(holdout)
    predictions = (probabilities >= bundle.threshold).astype(int)
    metrics = calculate_binary_metrics(
        holdout["TARGET"].astype(int),
        probabilities,
        threshold=bundle.threshold,
    )
    results = pd.DataFrame(
        {
            "SK_ID_CURR": holdout["SK_ID_CURR"].to_numpy(),
            "TARGET": holdout["TARGET"].astype(int).to_numpy(),
            "default_probability": probabilities,
            "predicted_class": predictions,
            "risk_category": bundle.assign_risk(probabilities),
        }
    )
    return metrics, results


def _feature_importance(
    bundle: CreditScoringModel,
) -> pd.DataFrame | None:
    """Извлечь нативную importance без тяжёлой SHAP-зависимости."""
    estimator = bundle.estimator
    if bundle.model_type == "catboost":
        values = np.asarray(
            getattr(estimator, "feature_importances_", []),
            dtype=float,
        )
        names = list(bundle.feature_names)
    elif (
        hasattr(estimator, "named_steps")
        and "preprocessor" in estimator.named_steps
        and "model" in estimator.named_steps
    ):
        preprocessor = estimator.named_steps["preprocessor"]
        model = estimator.named_steps["model"]
        if not hasattr(preprocessor, "get_feature_names_out") or not hasattr(
            model,
            "coef_",
        ):
            return None
        names = [str(name) for name in preprocessor.get_feature_names_out()]
        values = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    else:
        return None
    if len(names) != len(values) or len(values) == 0:
        return None
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def _save_figures(
    evaluation: pd.DataFrame,
    metrics: dict[str, Any],
    importance: pd.DataFrame | None,
    *,
    figures_dir: Path,
) -> None:
    """Сохранить confusion matrix, calibration, score и importance plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    confusion = np.array(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ]
    )
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(confusion, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{confusion[row, column]:,}",
                ha="center",
                va="center",
            )
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xlabel="Predicted class",
        ylabel="Actual class",
        title=f"Confusion matrix (threshold={metrics['threshold']:.3f})",
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    fraction_positive, mean_predicted = calibration_curve(
        evaluation["TARGET"],
        evaluation["default_probability"],
        n_bins=10,
        strategy="quantile",
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(mean_predicted, fraction_positive, marker="o", label="model")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="ideal")
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed default rate",
        title=f"Calibration curve (Brier={metrics['brier_score']:.4f})",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "calibration_curve.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    for target_value, color in [(0, "#4C78A8"), (1, "#E45756")]:
        axis.hist(
            evaluation.loc[
                evaluation["TARGET"].eq(target_value),
                "default_probability",
            ],
            bins=40,
            density=True,
            alpha=0.55,
            color=color,
            label=f"TARGET={target_value}",
        )
    axis.axvline(
        metrics["threshold"],
        color="black",
        linestyle="--",
        label="class threshold",
    )
    axis.set(
        xlabel="Predicted probability / score",
        ylabel="Density",
        title="Predicted default score distribution",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "score_distribution.png", dpi=150)
    plt.close(figure)

    if importance is not None and not importance.empty:
        top = importance.head(20).sort_values("importance")
        figure, axis = plt.subplots(figsize=(8, 7))
        axis.barh(top["feature"], top["importance"], color="#4C78A8")
        axis.set(
            xlabel="Absolute coefficient / feature importance",
            title="Top-20 model features",
        )
        figure.tight_layout()
        figure.savefig(figures_dir / "feature_importance.png", dpi=150)
        plt.close(figure)


def save_evaluation_artifacts(
    bundle: CreditScoringModel,
    metrics: dict[str, Any],
    evaluation: pd.DataFrame,
    *,
    tables_dir: Path = TABLES_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    """Сохранить компактные таблицы, error analysis и диагностические plots."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_row = {
        "model_type": bundle.model_type,
        "feature_set": bundle.feature_set,
        "is_smoke": bool(bundle.metadata.get("is_smoke", False)),
        **metrics,
    }
    pd.DataFrame([metrics_row]).to_csv(
        tables_dir / "final_evaluation.csv",
        index=False,
    )
    write_json_atomically(
        metrics_row,
        tables_dir / "final_evaluation.json",
    )
    pd.DataFrame(
        [
            {
                "actual": 0,
                "predicted_0": metrics["true_negative"],
                "predicted_1": metrics["false_positive"],
            },
            {
                "actual": 1,
                "predicted_0": metrics["false_negative"],
                "predicted_1": metrics["true_positive"],
            },
        ]
    ).to_csv(tables_dir / "confusion_matrix.csv", index=False)
    evaluation.loc[
        evaluation["TARGET"].ne(evaluation["predicted_class"])
    ].sort_values(
        "default_probability",
        ascending=False,
    ).to_csv(tables_dir / "error_analysis.csv", index=False)
    (
        evaluation.groupby("risk_category", observed=True)
        .agg(
            clients=("SK_ID_CURR", "size"),
            observed_default_rate=("TARGET", "mean"),
            mean_score=("default_probability", "mean"),
        )
        .reset_index()
        .to_csv(tables_dir / "risk_segments.csv", index=False)
    )
    importance = _feature_importance(bundle)
    if importance is not None:
        importance.to_csv(
            tables_dir / "feature_importance.csv",
            index=False,
        )
    _save_figures(
        evaluation,
        metrics,
        importance,
        figures_dir=figures_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    """Создать parser команды оценки."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="application_train.csv; по умолчанию ищется в data/raw",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--split",
        type=Path,
        default=CLIENT_SPLIT_PATH,
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=TABLES_DIR,
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Загрузить модель и получить независимую holdout-оценку."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    arguments = build_parser().parse_args(argv)
    model_path = resolve_model_path(arguments.model)
    bundle = load_model_bundle(model_path)
    input_path = (
        arguments.input
        if arguments.input is not None
        else find_data_file("application_train.csv")
    )
    application = read_application_csv(input_path, require_target=True)
    modeling_data = assemble_modeling_data(
        application,
        feature_set=bundle.feature_set,
    )
    split_path = arguments.split.expanduser().resolve()
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Client split не найден: {split_path}. "
            "Сначала выполните python3 -m src.train."
        )
    client_split = pd.read_csv(split_path)
    metrics, evaluation = evaluate_bundle(
        bundle,
        modeling_data,
        client_split,
    )
    save_evaluation_artifacts(
        bundle,
        metrics,
        evaluation,
        tables_dir=arguments.tables_dir,
        figures_dir=arguments.figures_dir,
    )
    LOGGER.info("Модель: %s", model_path)
    LOGGER.info(
        "Holdout ROC-AUC=%.6f, AP=%.6f, Brier=%.6f, KS=%.6f",
        metrics["roc_auc"],
        metrics["average_precision"],
        metrics["brier_score"],
        metrics["ks_statistic"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
