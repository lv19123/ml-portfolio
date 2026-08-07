"""Readable popularity and purchase-history recommendation baselines."""

from __future__ import annotations

import pandas as pd


def _validate_history(history: pd.DataFrame) -> None:
    required = {"customer_id", "article_id", "t_dat"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing required columns: {sorted(missing)}")
    if history.empty:
        raise ValueError("history must not be empty")


def popular_items(history: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    """Rank items by transaction count using history only."""
    _validate_history(history)
    if limit <= 0:
        raise ValueError("limit must be positive")
    result = (
        history.groupby("article_id", as_index=False)
        .size()
        .rename(columns={"size": "popularity_score"})
        .sort_values(
            ["popularity_score", "article_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(limit)
        .reset_index(drop=True)
    )
    result["popularity_rank"] = result.index + 1
    return result


def personal_history_candidates(
    history: pd.DataFrame,
    limit: int = 20,
) -> pd.DataFrame:
    """Return frequent personal-history pairs with score and rank."""
    _validate_history(history)
    pairs = (
        history.groupby(["customer_id", "article_id"], as_index=False)
        .agg(personal_history_score=("article_id", "size"), last_purchase=("t_dat", "max"))
        .sort_values(
            ["customer_id", "personal_history_score", "last_purchase", "article_id"],
            ascending=[True, False, False, True],
        )
    )
    pairs["personal_history_rank"] = pairs.groupby("customer_id").cumcount() + 1
    return pairs.loc[
        pairs["personal_history_rank"] <= limit,
        [
            "customer_id",
            "article_id",
            "personal_history_score",
            "personal_history_rank",
        ],
    ].reset_index(drop=True)


__all__ = [
    "personal_history_candidates",
    "popular_items",
]
