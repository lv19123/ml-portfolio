"""Batch scoring pipeline that loads trained artifacts instead of retraining."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fashion_recommender.als import build_matrix_with_mappings, generate_als_candidates
from fashion_recommender.baselines import personal_history_candidates, popular_items
from fashion_recommender.candidates import merge_candidate_sources, popularity_candidates
from fashion_recommender.content_based import (
    ContentArtifacts,
    build_user_profiles,
    generate_content_candidates,
    seen_items_by_user,
)
from fashion_recommender.data import load_articles, load_customers, load_transactions
from fashion_recommender.features import build_candidate_features
from fashion_recommender.persistence import (
    load_als_model,
    load_catboost_model,
    load_content_artifacts,
    load_json,
    save_json,
)
from fashion_recommender.ranking import build_top_k_recommendations, predict_purchase_scores


def run_batch_inference(
    project_root: str | Path,
    customer_ids: Iterable[str] | None = None,
    recommendation_size: int = 12,
    user_batch_size: int = 2_000,
    collect_results: bool = True,
) -> pd.DataFrame:
    """Load saved models, score current-history candidates, and write Parquet.

    Users are processed in controlled batches and written incrementally to one
    Parquet file.  Set ``collect_results=False`` for a full run so only one
    candidate/feature batch is kept in memory at a time.
    """
    if not 1 <= recommendation_size <= 50:
        raise ValueError("recommendation_size must be between 1 and 50")
    if user_batch_size <= 0:
        raise ValueError("user_batch_size must be positive")
    root = Path(project_root).expanduser().resolve()
    raw_dir = root / "data" / "raw"
    model_dir = root / "models"
    transactions = load_transactions(raw_dir / "transactions_train.csv")
    articles = load_articles(raw_dir / "articles.csv")
    customers = load_customers(raw_dir / "customers.csv")
    users = (
        transactions["customer_id"].drop_duplicates().astype(str).tolist()
        if customer_ids is None
        else [str(customer_id) for customer_id in customer_ids]
    )

    popular = popular_items(transactions, limit=100)
    personal_frame = personal_history_candidates(transactions, limit=20)

    als_model = load_als_model(model_dir / "als_model.npz")
    als_users = load_json(model_dir / "mappings" / "als_user_ids.json")
    als_items = load_json(model_dir / "mappings" / "als_article_ids.json")
    interactions = build_matrix_with_mappings(transactions, als_users, als_items)
    stored_content = load_content_artifacts(model_dir)
    content_config = stored_content.get("config", {})
    content = ContentArtifacts(
        encoder=stored_content["encoder"],
        article_feature_matrix=stored_content["article_feature_matrix"],
        article_to_index={
            article_id: index
            for index, article_id in enumerate(stored_content["article_ids"])
        },
        index_to_article=stored_content["article_ids"],
        feature_columns=content_config.get("feature_columns", []),
    )
    profiles = build_user_profiles(
        transactions,
        content,
        decay_days=float(content_config.get("decay_days", 30.0)),
    )
    all_seen_items = seen_items_by_user(transactions)
    reference_date = transactions["t_dat"].max() + pd.Timedelta(days=1)
    feature_columns = load_json(model_dir / "feature_columns.json")
    categorical_features = load_json(model_dir / "categorical_features.json")
    catboost_model = load_catboost_model(model_dir / "catboost_recommender.cbm")
    output_path = root / "artifacts" / "final_recommendations.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".parquet.in_progress")
    writer: pq.ParquetWriter | None = None
    collected: list[pd.DataFrame] = []
    total_rows = 0

    try:
        for batch_start in range(0, len(users), user_batch_size):
            user_batch = users[batch_start : batch_start + user_batch_size]
            als_frame = generate_als_candidates(
                als_model,
                interactions,
                user_batch,
                limit=150,
            )
            content_frame = generate_content_candidates(
                profiles,
                content,
                user_batch,
                seen_items=all_seen_items,
                fallback_items=popular["article_id"].tolist(),
                limit=50,
            )
            history_frame = personal_frame.loc[
                personal_frame["customer_id"].isin(user_batch)
            ]
            popularity_frame = popularity_candidates(user_batch, popular, limit=30)
            candidates = merge_candidate_sources(
                als=als_frame,
                content_based=content_frame,
                personal_history=history_frame,
                popularity=popularity_frame,
                limit_per_user=250,
            )
            features = build_candidate_features(
                candidates,
                transactions,
                articles,
                customers,
                reference_date=reference_date,
            )
            scored = predict_purchase_scores(
                catboost_model,
                features,
                feature_columns,
                categorical_features,
            )
            recommendations = build_top_k_recommendations(
                scored,
                popular["article_id"].tolist(),
                k=recommendation_size,
                customer_ids=user_batch,
            )
            recommendations["customer_id"] = recommendations["customer_id"].astype(str)
            recommendations["article_id"] = recommendations["article_id"].astype(str)
            recommendations["rank"] = recommendations["rank"].astype("int16")
            recommendations["score"] = recommendations["score"].astype(float)
            table = pa.Table.from_pandas(recommendations, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema)
            writer.write_table(table)
            total_rows += len(recommendations)
            if collect_results:
                collected.append(recommendations)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError("no users were provided for batch inference")
    temporary_path.replace(output_path)
    metadata_path = model_dir / "model_metadata.json"
    metadata = load_json(metadata_path)
    metadata["serving_artifact_users"] = len(users)
    metadata["serving_artifact_rows"] = total_rows
    metadata["serving_generated_at"] = pd.Timestamp.now(tz="UTC")
    save_json(metadata, metadata_path)
    if collect_results:
        return pd.concat(collected, ignore_index=True)
    return pd.DataFrame(columns=["customer_id", "article_id", "rank", "score"])


__all__ = ["run_batch_inference"]
