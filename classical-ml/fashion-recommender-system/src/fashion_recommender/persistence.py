"""Small, explicit persistence helpers for trained models and recommendations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz

from fashion_recommender.data import normalize_article_ids


def _existing_file(path: str | Path) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Artifact file does not exist: {file_path}")
    return file_path


def _prepare_parent(path: str | Path) -> Path:
    file_path = Path(path).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(value: Any, path: str | Path) -> Path:
    file_path = _prepare_parent(path)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=_json_default)
    return file_path


def load_json(path: str | Path) -> Any:
    file_path = _existing_file(path)
    with file_path.open(encoding="utf-8") as file:
        return json.load(file)


def save_als_model(model, path: str | Path) -> Path:
    file_path = _prepare_parent(path)
    model.save(str(file_path))
    actual_path = file_path if file_path.exists() else Path(f"{file_path}.npz")
    return actual_path


def load_als_model(path: str | Path):
    file_path = _existing_file(path)
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError as error:
        raise ImportError("Loading ALS requires the 'implicit' package") from error
    # ``implicit.als.AlternatingLeastSquares`` is a backend-selecting factory
    # in implicit 0.7, while ``load`` belongs to the selected CPU/GPU class.
    model_class = type(AlternatingLeastSquares())
    return model_class.load(str(file_path))


def save_content_artifacts(
    encoder,
    article_feature_matrix,
    article_ids: list[str],
    model_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    directory = Path(model_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mappings_dir = directory / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = directory / "content_encoder.joblib"
    matrix_path = directory / "article_feature_matrix.npz"
    mapping_path = mappings_dir / "content_article_ids.json"
    joblib.dump(encoder, encoder_path)
    save_npz(matrix_path, article_feature_matrix)
    save_json(article_ids, mapping_path)
    paths = {
        "encoder": encoder_path,
        "matrix": matrix_path,
        "article_ids": mapping_path,
    }
    if config is not None:
        config_path = directory / "content_config.json"
        save_json(config, config_path)
        paths["config"] = config_path
    return paths


def load_content_artifacts(model_dir: str | Path) -> dict[str, Any]:
    directory = Path(model_dir).expanduser().resolve()
    encoder_path = _existing_file(directory / "content_encoder.joblib")
    matrix_path = _existing_file(directory / "article_feature_matrix.npz")
    mapping_path = _existing_file(directory / "mappings" / "content_article_ids.json")
    result = {
        "encoder": joblib.load(encoder_path),
        "article_feature_matrix": load_npz(matrix_path).tocsr(),
        "article_ids": load_json(mapping_path),
    }
    config_path = directory / "content_config.json"
    if config_path.is_file():
        result["config"] = load_json(config_path)
    return result


def save_catboost_model(model, path: str | Path) -> Path:
    file_path = _prepare_parent(path)
    model.save_model(str(file_path), format="cbm")
    return file_path


def load_catboost_model(path: str | Path):
    file_path = _existing_file(path)
    try:
        from catboost import CatBoostClassifier
    except ImportError as error:
        raise ImportError("Loading CatBoost requires the 'catboost' package") from error
    model = CatBoostClassifier()
    model.load_model(str(file_path), format="cbm")
    return model


def save_recommendations(
    recommendations: pd.DataFrame,
    path: str | Path,
) -> Path:
    required = ["customer_id", "article_id", "rank", "score"]
    missing = set(required) - set(recommendations.columns)
    if missing:
        raise ValueError(f"recommendations is missing columns: {sorted(missing)}")
    result = recommendations.loc[:, required].copy()
    result["customer_id"] = result["customer_id"].astype("string")
    result["article_id"] = normalize_article_ids(result["article_id"])
    result["rank"] = pd.to_numeric(result["rank"], errors="raise").astype("int16")
    result["score"] = pd.to_numeric(result["score"], errors="raise").astype(float)
    if result.duplicated(["customer_id", "article_id"]).any():
        raise ValueError("recommendations contains duplicate user-item pairs")
    if (result["rank"] <= 0).any():
        raise ValueError("recommendation ranks must be positive")
    file_path = _prepare_parent(path)
    result.to_parquet(file_path, index=False)
    return file_path


def load_recommendations(path: str | Path) -> pd.DataFrame:
    file_path = _existing_file(path)
    result = pd.read_parquet(file_path)
    required = {"customer_id", "article_id", "rank", "score"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"recommendation artifact is missing: {sorted(missing)}")
    result["customer_id"] = result["customer_id"].astype("string")
    result["article_id"] = normalize_article_ids(result["article_id"])
    return result.sort_values(["customer_id", "rank"]).reset_index(drop=True)


__all__ = [
    "load_als_model",
    "load_catboost_model",
    "load_content_artifacts",
    "load_json",
    "load_recommendations",
    "save_als_model",
    "save_catboost_model",
    "save_content_artifacts",
    "save_json",
    "save_recommendations",
]
