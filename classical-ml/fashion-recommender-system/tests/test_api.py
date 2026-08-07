from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from api.service import RecommendationService
from fashion_recommender.persistence import save_json, save_recommendations


def api_client(tmp_path: Path) -> TestClient:
    recommendations = pd.DataFrame(
        {
            "customer_id": ["known", "known"],
            "article_id": ["0000000001", "0000000002"],
            "rank": [1, 2],
            "score": [0.9, 0.8],
        }
    )
    recommendations_path = save_recommendations(recommendations, tmp_path / "final.parquet")
    popular_path = save_json(["0000000003", "0000000004"], tmp_path / "popular.json")
    metadata_path = save_json(
        {"architecture": "candidate generation + CatBoost", "prediction_horizon_days": 7},
        tmp_path / "metadata.json",
    )
    service = RecommendationService(recommendations_path, popular_path, metadata_path)
    return TestClient(create_app(service=service))


def test_health_reports_loaded_artifact(tmp_path: Path) -> None:
    response = api_client(tmp_path).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "recommendations_loaded": True}


def test_known_user_receives_personalized_items_and_k(tmp_path: Path) -> None:
    response = api_client(tmp_path).get("/recommend/known?k=1")
    assert response.status_code == 200
    assert response.json()["source"] == "personalized"
    assert response.json()["recommendations"] == [{"article_id": "0000000001", "rank": 1}]


def test_unknown_user_receives_popularity_fallback(tmp_path: Path) -> None:
    response = api_client(tmp_path).get("/recommend/unknown?k=2")
    assert response.status_code == 200
    assert response.json()["source"] == "popularity_fallback"
    assert len(response.json()["recommendations"]) == 2


def test_k_is_limited_to_safe_range(tmp_path: Path) -> None:
    assert api_client(tmp_path).get("/recommend/known?k=0").status_code == 422
    assert api_client(tmp_path).get("/recommend/known?k=51").status_code == 422


def test_model_info_returns_saved_metadata(tmp_path: Path) -> None:
    response = api_client(tmp_path).get("/model-info")
    assert response.status_code == 200
    assert response.json()["prediction_horizon_days"] == 7
