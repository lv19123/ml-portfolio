"""One-hot content recommendations based on weighted user profiles."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder, normalize


ITEM_FEATURE_COLUMNS = [
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "department_name",
    "section_name",
    "garment_group_name",
]


@dataclass(frozen=True)
class ContentArtifacts:
    encoder: OneHotEncoder
    article_feature_matrix: csr_matrix
    article_to_index: dict[str, int]
    index_to_article: list[str]
    feature_columns: list[str]


@dataclass(frozen=True)
class UserProfiles:
    matrix: csr_matrix
    user_to_index: dict[str, int]
    index_to_user: list[str]


def fit_content_encoder(
    articles: pd.DataFrame,
    feature_columns: Sequence[str] = ITEM_FEATURE_COLUMNS,
) -> ContentArtifacts:
    """Fit a sparse ``OneHotEncoder`` and transform item attributes."""
    required = {"article_id", *feature_columns}
    missing = required - set(articles.columns)
    if missing:
        raise ValueError(f"articles is missing required columns: {sorted(missing)}")
    if articles["article_id"].duplicated().any():
        raise ValueError("articles must contain one row per article_id")

    clean = articles.loc[:, ["article_id", *feature_columns]].copy()
    clean[list(feature_columns)] = (
        clean[list(feature_columns)].fillna("Unknown").astype(str)
    )
    clean["article_id"] = clean["article_id"].astype(str)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    matrix = encoder.fit_transform(clean[list(feature_columns)]).tocsr().astype(np.float32)
    index_to_article = clean["article_id"].tolist()
    article_to_index = {
        article_id: index for index, article_id in enumerate(index_to_article)
    }
    return ContentArtifacts(
        encoder=encoder,
        article_feature_matrix=matrix,
        article_to_index=article_to_index,
        index_to_article=index_to_article,
        feature_columns=list(feature_columns),
    )


def calculate_recency_weight(days_since_purchase, decay_days: float = 30.0):
    if decay_days <= 0:
        raise ValueError("decay_days must be positive")
    days = np.asarray(days_since_purchase, dtype=float)
    if np.any(days < 0):
        raise ValueError("days_since_purchase must be non-negative")
    return np.exp(-days / decay_days)


def calculate_frequency_weight(purchase_count):
    counts = np.asarray(purchase_count, dtype=float)
    if np.any(counts <= 0):
        raise ValueError("purchase_count must be positive")
    return 1.0 + np.log1p(counts)


def build_user_profiles(
    history: pd.DataFrame,
    content: ContentArtifacts,
    reference_date: pd.Timestamp | None = None,
    decay_days: float = 30.0,
) -> UserProfiles:
    """Create sparse weighted-mean user profiles from purchases before cutoff."""
    required = {"customer_id", "article_id", "t_dat"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing required columns: {sorted(missing)}")
    if history.empty:
        raise ValueError("history must not be empty")
    if not pd.api.types.is_datetime64_any_dtype(history["t_dat"]):
        raise TypeError("history['t_dat'] must have a datetime dtype")

    reference = (
        pd.Timestamp(history["t_dat"].max()) + pd.Timedelta(days=1)
        if reference_date is None
        else pd.Timestamp(reference_date)
    )
    pairs = (
        history.groupby(["customer_id", "article_id"], as_index=False)
        .agg(purchase_count=("article_id", "size"), last_purchase=("t_dat", "max"))
    )
    pairs = pairs.loc[
        pairs["article_id"].astype(str).isin(content.article_to_index)
    ].copy()
    if pairs.empty:
        raise ValueError("history does not contain articles known to the encoder")
    days = (reference - pairs["last_purchase"]).dt.days.clip(lower=0)
    pairs["purchase_weight"] = calculate_recency_weight(
        days,
        decay_days,
    ) * calculate_frequency_weight(pairs["purchase_count"])

    index_to_user = sorted(pairs["customer_id"].astype(str).unique())
    user_to_index = {value: index for index, value in enumerate(index_to_user)}
    rows = pairs["customer_id"].astype(str).map(user_to_index).to_numpy()
    columns = pairs["article_id"].astype(str).map(content.article_to_index).to_numpy()
    weights = pairs["purchase_weight"].to_numpy(dtype=np.float32)
    weighted_interactions = csr_matrix(
        (weights, (rows, columns)),
        shape=(len(index_to_user), len(content.index_to_article)),
        dtype=np.float32,
    )
    profiles = (weighted_interactions @ content.article_feature_matrix).tocsr()
    weight_sums = np.asarray(weighted_interactions.sum(axis=1)).ravel()
    profiles = profiles.multiply((1.0 / weight_sums)[:, np.newaxis]).tocsr()
    return UserProfiles(
        matrix=profiles,
        user_to_index=user_to_index,
        index_to_user=index_to_user,
    )


def generate_content_candidates(
    profiles: UserProfiles,
    content: ContentArtifacts,
    customer_ids: Iterable[str] | None = None,
    seen_items: dict[str, set[str]] | None = None,
    fallback_items: Sequence[str] | None = None,
    limit: int = 50,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Rank articles by cosine similarity without an item-item dense matrix."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    users = (
        profiles.index_to_user
        if customer_ids is None
        else [str(customer_id) for customer_id in customer_ids]
    )
    normalized_profiles = normalize(profiles.matrix, norm="l2", axis=1)
    normalized_items = normalize(content.article_feature_matrix, norm="l2", axis=1)
    fallback = list(dict.fromkeys(str(item) for item in (fallback_items or [])))
    completion_pool = list(dict.fromkeys([*fallback, *content.index_to_article]))
    seen_items = seen_items or {}
    rows: list[dict[str, object]] = []

    for batch_start in range(0, len(users), batch_size):
        user_batch = users[batch_start : batch_start + batch_size]
        known_positions = [
            position
            for position, customer_id in enumerate(user_batch)
            if customer_id in profiles.user_to_index
        ]
        known_indices = [
            profiles.user_to_index[user_batch[position]]
            for position in known_positions
        ]
        similarities = (
            normalized_profiles[known_indices].dot(normalized_items.T).tocsr()
            if known_indices
            else None
        )
        similarity_row_by_position = {
            position: row_number
            for row_number, position in enumerate(known_positions)
        }

        for position, customer_id in enumerate(user_batch):
            ranked: list[tuple[str, float]] = []
            similarity_row = similarity_row_by_position.get(position)
            if similarities is not None and similarity_row is not None:
                row_start = similarities.indptr[similarity_row]
                row_end = similarities.indptr[similarity_row + 1]
                item_indices = similarities.indices[row_start:row_end]
                similarity_scores = similarities.data[row_start:row_end]
                user_seen_indices = {
                    content.article_to_index[article_id]
                    for article_id in seen_items.get(customer_id, set())
                    if article_id in content.article_to_index
                }
                if user_seen_indices:
                    keep_mask = ~np.isin(
                        item_indices,
                        np.fromiter(user_seen_indices, dtype=np.int64),
                    )
                    item_indices = item_indices[keep_mask]
                    similarity_scores = similarity_scores[keep_mask]
                number_to_select = min(limit, len(similarity_scores))
                if number_to_select:
                    if len(similarity_scores) > number_to_select:
                        top_positions = np.argpartition(
                            similarity_scores,
                            -number_to_select,
                        )[-number_to_select:]
                    else:
                        top_positions = np.arange(number_to_select)
                    candidates = [
                        (
                            content.index_to_article[int(item_indices[top_position])],
                            float(similarity_scores[top_position]),
                        )
                        for top_position in top_positions
                    ]
                    ranked = sorted(
                        candidates,
                        key=lambda pair: (-pair[1], pair[0]),
                    )

            present = {article_id for article_id, _ in ranked}
            # Sparse multiplication omits exact zero similarities.  A short
            # list is completed by popularity and then remaining catalog items.
            if len(ranked) < limit:
                for article_id in completion_pool:
                    if (
                        article_id not in present
                        and article_id not in seen_items.get(customer_id, set())
                    ):
                        ranked.append((article_id, 0.0))
                        present.add(article_id)
                    if len(ranked) == limit:
                        break
            for rank, (article_id, score) in enumerate(ranked[:limit], 1):
                rows.append(
                    {
                        "customer_id": customer_id,
                        "article_id": article_id,
                        "content_similarity_score": score,
                        "content_rank": rank,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "customer_id",
            "article_id",
            "content_similarity_score",
            "content_rank",
        ],
    )


def seen_items_by_user(history: pd.DataFrame) -> dict[str, set[str]]:
    return (
        history.groupby("customer_id")["article_id"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )


__all__ = [
    "ContentArtifacts",
    "ITEM_FEATURE_COLUMNS",
    "UserProfiles",
    "build_user_profiles",
    "calculate_frequency_weight",
    "calculate_recency_weight",
    "fit_content_encoder",
    "generate_content_candidates",
    "seen_items_by_user",
]
