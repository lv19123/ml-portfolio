import numpy as np
import pandas as pd
import pytest

from src.experiment_tracking import (
    MODEL_METRICS_COLUMNS,
    save_experiment_result,
)


def make_result(
    *,
    experiment="application_bureau",
    model="CatBoostClassifier",
    cv_roc_auc=0.75,
    holdout_roc_auc=None,
    holdout_pr_auc=None,
):
    return {
        "experiment": experiment,
        "notebook": "03_bureau_features.ipynb",
        "model": model,
        "feature_set": "application + bureau",
        "source_tables": "application_train.csv, bureau.csv",
        "device": "CPU",
        "n_train": 80,
        "n_holdout": 20,
        "n_features": 50,
        "cv_folds": 3,
        "best_iteration": 100,
        "cv_roc_auc": cv_roc_auc,
        "cv_roc_auc_std": 0.01,
        "cv_pr_auc": 0.30,
        "cv_pr_auc_std": 0.02,
        "holdout_roc_auc": holdout_roc_auc,
        "holdout_pr_auc": holdout_pr_auc,
    }


def test_creates_new_csv_with_expected_columns(tmp_path):
    path = tmp_path / "tables" / "model_metrics.csv"

    saved = save_experiment_result(make_result(), path)

    assert path.is_file()
    assert saved.columns.tolist() == MODEL_METRICS_COLUMNS
    assert pd.read_csv(path).columns.tolist() == MODEL_METRICS_COLUMNS


def test_accepts_mapping_in_different_key_order(tmp_path):
    path = tmp_path / "model_metrics.csv"
    result = make_result()
    reversed_result = {
        key: result[key]
        for key in reversed(list(result))
    }

    saved = save_experiment_result(
        reversed_result,
        path,
    )

    assert saved.columns.tolist() == MODEL_METRICS_COLUMNS


def test_adds_new_row_and_sorts_by_experiment_order(tmp_path):
    path = tmp_path / "model_metrics.csv"
    save_experiment_result(
        make_result(experiment="application_credit_card"),
        path,
    )

    saved = save_experiment_result(
        make_result(
            experiment="application_logistic",
            model="LogisticRegression",
        ),
        path,
    )

    assert saved["experiment"].tolist() == [
        "application_logistic",
        "application_credit_card",
    ]


def test_repeated_result_is_replaced_without_duplicate(tmp_path):
    path = tmp_path / "model_metrics.csv"
    save_experiment_result(make_result(cv_roc_auc=0.70), path)

    saved = save_experiment_result(
        make_result(cv_roc_auc=0.81),
        path,
    )

    assert len(saved) == 1
    assert saved.loc[0, "cv_roc_auc"] == pytest.approx(0.81)


def test_same_experiment_with_other_model_is_preserved(tmp_path):
    path = tmp_path / "model_metrics.csv"
    save_experiment_result(
        make_result(
            experiment="application_logistic",
            model="LogisticRegression",
        ),
        path,
    )

    saved = save_experiment_result(
        make_result(
            experiment="application_logistic",
            model="CatBoostClassifier",
        ),
        path,
    )

    assert len(saved) == 2


def test_empty_holdout_metrics_are_saved_as_nan(tmp_path):
    path = tmp_path / "model_metrics.csv"

    save_experiment_result(make_result(), path)
    reloaded = pd.read_csv(path)

    assert np.isnan(reloaded.loc[0, "holdout_roc_auc"])
    assert np.isnan(reloaded.loc[0, "holdout_pr_auc"])


def test_incorrect_existing_schema_raises_error(tmp_path):
    path = tmp_path / "model_metrics.csv"
    pd.DataFrame(
        {
            "experiment": ["legacy"],
            "score": [0.5],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Некорректная схема"):
        save_experiment_result(make_result(), path)


def test_incorrect_result_schema_raises_error(tmp_path):
    path = tmp_path / "model_metrics.csv"
    incomplete_result = make_result()
    incomplete_result.pop("device")

    with pytest.raises(ValueError, match="Некорректная схема"):
        save_experiment_result(incomplete_result, path)
