import pandas as pd
import pytest

from src.model_bundle import CreditScoringModel
from src.predict import predict_dataframe
from src.preprocessing import make_logistic_pipeline


def prediction_bundle():
    train = pd.DataFrame(
        {
            "AMT_CREDIT": [50.0, 100.0, 150.0, 250.0, 300.0, 350.0],
            "NAME_CONTRACT_TYPE": ["Cash", "Cash", "Card", "Card", "Cash", "Card"],
        }
    )
    target = pd.Series([0, 0, 0, 1, 1, 1])
    estimator = make_logistic_pipeline()
    estimator.fit(train, target)
    return CreditScoringModel(
        estimator=estimator,
        feature_names=("AMT_CREDIT", "NAME_CONTRACT_TYPE"),
        model_type="logistic",
    )


def test_prediction_output_has_required_columns_and_rows():
    application = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002],
            "AMT_CREDIT": [90.0, 320.0],
            "NAME_CONTRACT_TYPE": ["Unknown contract", "Cash"],
        }
    )

    predictions = predict_dataframe(application, prediction_bundle())

    assert predictions.columns.tolist() == [
        "SK_ID_CURR",
        "default_probability",
        "predicted_class",
        "risk_category",
    ]
    assert len(predictions) == len(application)
    assert predictions["default_probability"].between(0, 1).all()
    assert set(predictions["predicted_class"]).issubset({0, 1})
    assert set(predictions["risk_category"]).issubset(
        {"low", "medium", "high"}
    )


def test_prediction_rejects_target_column():
    application = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
            "TARGET": [0],
            "AMT_CREDIT": [90.0],
            "NAME_CONTRACT_TYPE": ["Cash"],
        }
    )

    with pytest.raises(ValueError, match="TARGET"):
        predict_dataframe(application, prediction_bundle())
