"""Combine recommendation sources into one user-item candidate table."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


SOURCE_COLUMNS = {
    "als": ("als_score", "als_rank", "from_als"),
    "content_based": (
        "content_similarity_score",
        "content_rank",
        "from_content_based",
    ),
    "personal_history": (
        "personal_history_score",
        "personal_history_rank",
        "from_personal_history",
    ),
    "popularity": (
        "popularity_score",
        "popularity_rank",
        "from_popularity",
    ),
}


def popularity_candidates(
    customer_ids: Iterable[str],
    popular_items: pd.DataFrame,
    limit: int = 30,
) -> pd.DataFrame:
    """Repeat a small global popularity list for the requested users."""
    required = {"article_id", "popularity_score", "popularity_rank"}
    missing = required - set(popular_items.columns)
    if missing:
        raise ValueError(f"popular_items is missing columns: {sorted(missing)}")
    top = popular_items.nsmallest(limit, "popularity_rank").copy()
    users = pd.DataFrame({"customer_id": [str(user) for user in customer_ids]})
    users["_key"] = 1
    top["_key"] = 1
    result = users.merge(top, on="_key", how="inner").drop(columns="_key")
    return result[["customer_id", "article_id", "popularity_score", "popularity_rank"]]


def merge_candidate_sources(
    *,
    als: pd.DataFrame | None = None,
    content_based: pd.DataFrame | None = None,
    personal_history: pd.DataFrame | None = None,
    popularity: pd.DataFrame | None = None,
    limit_per_user: int | None = None,
) -> pd.DataFrame:
    """Outer-merge sources and keep exactly one row per user-item pair."""
    frames = {
        "als": als,
        "content_based": content_based,
        "personal_history": personal_history,
        "popularity": popularity,
    }
    active = {name: frame for name, frame in frames.items() if frame is not None}
    if not active:
        raise ValueError("at least one candidate source is required")

    base_parts = []
    prepared: dict[str, pd.DataFrame] = {}
    for name, frame in active.items():
        assert frame is not None
        score_column, rank_column, _ = SOURCE_COLUMNS[name]
        required = {"customer_id", "article_id", score_column, rank_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} candidates are missing: {sorted(missing)}")
        clean = frame.loc[:, list(required)].copy()
        clean["customer_id"] = clean["customer_id"].astype(str)
        clean["article_id"] = clean["article_id"].astype(str)
        clean = (
            clean.sort_values(rank_column)
            .drop_duplicates(["customer_id", "article_id"], keep="first")
        )
        prepared[name] = clean
        base_parts.append(clean[["customer_id", "article_id"]])

    result = pd.concat(base_parts, ignore_index=True).drop_duplicates()
    for name, clean in prepared.items():
        score_column, rank_column, flag_column = SOURCE_COLUMNS[name]
        result = result.merge(clean, on=["customer_id", "article_id"], how="left")
        result[flag_column] = result[rank_column].notna().astype("int8")

    for name, (score_column, rank_column, flag_column) in SOURCE_COLUMNS.items():
        if score_column not in result:
            result[score_column] = 0.0
            result[rank_column] = 0
            result[flag_column] = np.int8(0)
        else:
            result[score_column] = result[score_column].fillna(0.0).astype(float)
            result[rank_column] = result[rank_column].fillna(0).astype("int32")

    flag_columns = [columns[2] for columns in SOURCE_COLUMNS.values()]
    result["number_of_candidate_sources"] = result[flag_columns].sum(axis=1).astype("int8")
    result["_rank_priority"] = sum(
        np.where(result[rank] > 0, 1.0 / result[rank], 0.0)
        for _, rank, _ in SOURCE_COLUMNS.values()
    )
    result = result.sort_values(
        ["customer_id", "number_of_candidate_sources", "_rank_priority", "article_id"],
        ascending=[True, False, False, True],
    )
    if limit_per_user is not None:
        if limit_per_user <= 0:
            raise ValueError("limit_per_user must be positive")
        result = result.loc[result.groupby("customer_id").cumcount() < limit_per_user]
    return result.drop(columns="_rank_priority").reset_index(drop=True)


__all__ = [
    "SOURCE_COLUMNS",
    "merge_candidate_sources",
    "popularity_candidates",
]
