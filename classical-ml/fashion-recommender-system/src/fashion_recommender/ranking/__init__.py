"""Reusable scoring helpers for the optional batch-inference layer."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd


def predict_purchase_scores(
    model,
    feature_table: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """Add CatBoost purchase probabilities to candidate identifiers."""
    prediction_data = feature_table.loc[:, feature_columns].copy()
    for column in categorical_features:
        prediction_data[column] = prediction_data[column].astype(str)
    result = feature_table.loc[:, ["customer_id", "article_id"]].copy()
    result["score"] = model.predict_proba(prediction_data)[:, 1]
    return result


def build_top_k_recommendations(
    scored_candidates: pd.DataFrame,
    popular_article_ids: Sequence[str],
    k: int = 12,
    customer_ids: Iterable[str] | None = None,
    score_column: str = "score",
) -> pd.DataFrame:
    """Sort candidates, remove repeats, and fill short lists by popularity."""
    if not 1 <= k <= 50:
        raise ValueError("k must be between 1 and 50")
    required = {"customer_id", "article_id", score_column}
    missing = required - set(scored_candidates.columns)
    if missing:
        raise ValueError(f"scored_candidates is missing: {sorted(missing)}")
    clean = scored_candidates.copy()
    clean["customer_id"] = clean["customer_id"].astype(str)
    clean["article_id"] = clean["article_id"].astype(str)
    clean = (
        clean.sort_values(
            ["customer_id", score_column, "article_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates(["customer_id", "article_id"])
    )
    by_user = {
        customer_id: list(zip(group["article_id"], group[score_column], strict=True))
        for customer_id, group in clean.groupby("customer_id", sort=False)
    }
    users = list(by_user) if customer_ids is None else [str(user) for user in customer_ids]
    fallback = list(dict.fromkeys(str(article_id) for article_id in popular_article_ids))
    rows = []
    for customer_id in users:
        recommendations = by_user.get(customer_id, [])
        chosen: list[tuple[str, float]] = []
        seen: set[str] = set()
        for article_id, score in recommendations:
            if article_id not in seen:
                chosen.append((article_id, float(score)))
                seen.add(article_id)
            if len(chosen) >= k:
                break
        for article_id in fallback:
            if article_id not in seen:
                chosen.append((article_id, 0.0))
                seen.add(article_id)
            if len(chosen) >= k:
                break
        rows.extend(
            {
                "customer_id": customer_id,
                "article_id": article_id,
                "rank": rank,
                "score": score,
            }
            for rank, (article_id, score) in enumerate(chosen[:k], 1)
        )
    return pd.DataFrame(rows, columns=["customer_id", "article_id", "rank", "score"])


__all__ = [
    "build_top_k_recommendations",
    "predict_purchase_scores",
]
