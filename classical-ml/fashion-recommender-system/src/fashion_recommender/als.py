"""Sparse-matrix preparation, ALS training, and ALS candidate generation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class InteractionMatrix:
    """A sparse user-item matrix and deterministic identifier mappings."""

    matrix: csr_matrix
    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    index_to_user: list[str]
    index_to_item: list[str]


def prepare_user_item_matrix(history: pd.DataFrame) -> InteractionMatrix:
    """Build a CSR matrix whose values are ``1 + log1p(purchase_count)``."""
    required = {"customer_id", "article_id"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing required columns: {sorted(missing)}")
    if history.empty:
        raise ValueError("history must not be empty")

    counts = (
        history.groupby(["customer_id", "article_id"], as_index=False)
        .size()
        .rename(columns={"size": "purchase_count"})
    )
    index_to_user = sorted(counts["customer_id"].astype(str).unique())
    index_to_item = sorted(counts["article_id"].astype(str).unique())
    user_to_index = {value: index for index, value in enumerate(index_to_user)}
    item_to_index = {value: index for index, value in enumerate(index_to_item)}

    rows = counts["customer_id"].astype(str).map(user_to_index).to_numpy()
    columns = counts["article_id"].astype(str).map(item_to_index).to_numpy()
    confidence = 1.0 + np.log1p(counts["purchase_count"].to_numpy(dtype=float))
    matrix = csr_matrix(
        (confidence.astype(np.float32), (rows, columns)),
        shape=(len(index_to_user), len(index_to_item)),
        dtype=np.float32,
    )
    return InteractionMatrix(
        matrix=matrix,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_user=index_to_user,
        index_to_item=index_to_item,
    )


def build_matrix_with_mappings(
    history: pd.DataFrame,
    index_to_user: list[str],
    index_to_item: list[str],
) -> InteractionMatrix:
    """Rebuild filter interactions while preserving saved ALS factor indices."""
    required = {"customer_id", "article_id"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing required columns: {sorted(missing)}")
    user_to_index = {value: index for index, value in enumerate(index_to_user)}
    item_to_index = {value: index for index, value in enumerate(index_to_item)}
    counts = (
        history.groupby(["customer_id", "article_id"], as_index=False)
        .size()
        .rename(columns={"size": "purchase_count"})
    )
    counts["customer_id"] = counts["customer_id"].astype(str)
    counts["article_id"] = counts["article_id"].astype(str)
    counts = counts.loc[
        counts["customer_id"].isin(user_to_index)
        & counts["article_id"].isin(item_to_index)
    ]
    rows = counts["customer_id"].map(user_to_index).to_numpy()
    columns = counts["article_id"].map(item_to_index).to_numpy()
    confidence = 1.0 + np.log1p(counts["purchase_count"].to_numpy(dtype=float))
    matrix = csr_matrix(
        (confidence.astype(np.float32), (rows, columns)),
        shape=(len(index_to_user), len(index_to_item)),
        dtype=np.float32,
    )
    return InteractionMatrix(
        matrix=matrix,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_user=index_to_user,
        index_to_item=index_to_item,
    )


def generate_als_candidates(
    model,
    interactions: InteractionMatrix,
    customer_ids: Iterable[str] | None = None,
    limit: int = 150,
    batch_size: int = 1000,
) -> pd.DataFrame:
    """Return ALS item scores and ranks for known users only."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    users = (
        list(interactions.user_to_index)
        if customer_ids is None
        else [str(customer_id) for customer_id in customer_ids]
    )
    known_users = [
        (customer_id, interactions.user_to_index[customer_id])
        for customer_id in users
        if customer_id in interactions.user_to_index
    ]
    rows: list[dict[str, object]] = []
    for start in range(0, len(known_users), batch_size):
        batch = known_users[start : start + batch_size]
        user_indices = np.asarray([user_index for _, user_index in batch])
        item_indices, scores = model.recommend(
            user_indices,
            interactions.matrix[user_indices],
            N=limit,
            filter_already_liked_items=True,
        )
        item_indices = np.atleast_2d(item_indices)
        scores = np.atleast_2d(scores)
        for row_number, (customer_id, _) in enumerate(batch):
            for rank, (item_index, score) in enumerate(
                zip(item_indices[row_number], scores[row_number], strict=True),
                start=1,
            ):
                rows.append(
                    {
                        "customer_id": customer_id,
                        "article_id": interactions.index_to_item[int(item_index)],
                        "als_score": float(score),
                        "als_rank": rank,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["customer_id", "article_id", "als_score", "als_rank"],
    )


__all__ = [
    "InteractionMatrix",
    "build_matrix_with_mappings",
    "generate_als_candidates",
    "prepare_user_item_matrix",
]
