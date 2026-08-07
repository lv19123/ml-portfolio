from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

from fashion_recommender.persistence import (
    load_als_model,
    load_catboost_model,
    load_content_artifacts,
    load_json,
    load_recommendations,
    save_als_model,
    save_catboost_model,
    save_content_artifacts,
    save_json,
    save_recommendations,
)
from fashion_recommender.als import prepare_user_item_matrix


def test_json_round_trip_and_missing_file(tmp_path: Path) -> None:
    path = save_json({"date": pd.Timestamp("2020-01-01"), "value": np.int64(2)}, tmp_path / "a.json")
    assert load_json(path) == {"date": "2020-01-01 00:00:00", "value": 2}
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_json(tmp_path / "missing.json")


def test_content_artifacts_round_trip(tmp_path: Path) -> None:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit([["red"], ["blue"]])
    matrix = csr_matrix([[1, 0], [0, 1]], dtype=np.float32)
    save_content_artifacts(
        encoder,
        matrix,
        ["0000000001", "0000000002"],
        tmp_path,
        {"decay_days": 30},
    )
    loaded = load_content_artifacts(tmp_path)
    assert loaded["article_feature_matrix"].shape == (2, 2)
    assert loaded["article_ids"] == ["0000000001", "0000000002"]
    assert loaded["config"]["decay_days"] == 30


def test_recommendations_parquet_round_trip_and_normalizes_id(tmp_path: Path) -> None:
    recommendations = pd.DataFrame(
        {
            "customer_id": ["u1", "u1"],
            "article_id": ["123", "42.0"],
            "rank": [1, 2],
            "score": [0.9, 0.8],
        }
    )
    path = save_recommendations(recommendations, tmp_path / "recommendations.parquet")
    loaded = load_recommendations(path)
    assert loaded["article_id"].tolist() == ["0000000123", "0000000042"]


def test_recommendations_reject_duplicates(tmp_path: Path) -> None:
    recommendations = pd.DataFrame(
        {"customer_id": ["u1", "u1"], "article_id": ["1", "1"], "rank": [1, 2], "score": [1, 0]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        save_recommendations(recommendations, tmp_path / "x.parquet")


def test_als_model_round_trip(tmp_path: Path) -> None:
    history = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "article_id": ["A", "B", "A", "C", "B", "C"],
        }
    )
    interactions = prepare_user_item_matrix(history)
    model = AlternatingLeastSquares(factors=2, iterations=2, random_state=1)
    model.fit(interactions.matrix, show_progress=False)
    path = save_als_model(model, tmp_path / "als_model.npz")
    loaded = load_als_model(path)
    assert loaded.user_factors.shape == model.user_factors.shape


def test_catboost_model_round_trip(tmp_path: Path) -> None:
    catboost = pytest.importorskip("catboost")
    model = catboost.CatBoostClassifier(
        iterations=2,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}), [0, 0, 1, 1])
    path = save_catboost_model(model, tmp_path / "model.cbm")
    loaded = load_catboost_model(path)
    assert loaded.predict_proba([[1.5]]).shape == (1, 2)
    save_als_model,
    save_catboost_model,
