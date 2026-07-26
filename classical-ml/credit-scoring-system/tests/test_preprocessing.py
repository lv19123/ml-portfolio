import numpy as np
import pandas as pd

from src.preprocessing import (
    infer_categorical_features,
    make_logistic_pipeline,
    prepare_catboost_features,
)


def training_frame():
    return pd.DataFrame(
        {
            "amount": [100.0, 150.0, np.nan, 300.0, 220.0, 80.0],
            "segment": ["A", "B", None, "A", "B", "A"],
        }
    )


def test_logistic_preprocessing_preserves_rows_and_unknown_categories():
    features = training_frame()
    target = pd.Series([0, 1, 0, 1, 1, 0])
    pipeline = make_logistic_pipeline()
    pipeline.fit(features, target)

    inference = pd.DataFrame(
        {
            "amount": [125.0, np.nan],
            "segment": ["UNSEEN", None],
        }
    )
    probabilities = pipeline.predict_proba(inference)

    assert probabilities.shape == (len(inference), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_catboost_preprocessing_preserves_rows_and_fills_categories():
    features = training_frame()
    categorical = infer_categorical_features(features)
    prepared = prepare_catboost_features(features, categorical)

    assert categorical == ["segment"]
    assert len(prepared) == len(features)
    assert prepared["segment"].isna().sum() == 0
    assert prepared.loc[2, "segment"] == "Unknown"
    assert prepared["segment"].dtype == object
