import numpy as np
import pandas as pd
import pytest

from src.model_bundle import (
    CreditScoringModel,
    load_model_bundle,
    save_model_bundle,
)
from src.preprocessing import make_logistic_pipeline


def fitted_bundle():
    features = pd.DataFrame(
        {
            "amount": [50.0, 80.0, 120.0, 180.0, 250.0, 320.0],
            "category": ["A", "A", "B", "B", "C", "C"],
        }
    )
    target = pd.Series([0, 0, 0, 1, 1, 1])
    estimator = make_logistic_pipeline()
    estimator.fit(features, target)
    return CreditScoringModel(
        estimator=estimator,
        feature_names=("amount", "category"),
        model_type="logistic",
        threshold=0.4,
        risk_cutoffs=(0.3, 0.7),
    )


def test_bundle_prediction_format_and_probability_range():
    bundle = fitted_bundle()
    inference = pd.DataFrame(
        {
            "amount": [90.0, 280.0],
            "category": ["UNSEEN", "C"],
        }
    )

    probabilities = bundle.predict_proba(inference)
    predictions = bundle.predict(inference)

    assert probabilities.shape == (2,)
    assert predictions.shape == (2,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert set(predictions).issubset({0, 1})


def test_saved_bundle_can_be_loaded(tmp_path):
    path = tmp_path / "nested" / "model.joblib"
    save_model_bundle(fitted_bundle(), path)

    loaded = load_model_bundle(path)

    assert isinstance(loaded, CreditScoringModel)
    assert loaded.feature_names == ("amount", "category")


def test_missing_model_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        load_model_bundle(tmp_path / "missing.joblib")


def test_bundle_rejects_missing_features():
    bundle = fitted_bundle()
    with pytest.raises(ValueError, match="category"):
        bundle.predict_proba(pd.DataFrame({"amount": [100.0]}))


def test_risk_categories_follow_saved_cutoffs():
    bundle = fitted_bundle()
    result = bundle.assign_risk(np.array([0.1, 0.5, 0.9]))

    assert result.tolist() == ["low", "medium", "high"]
