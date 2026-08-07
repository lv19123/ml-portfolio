"""Leakage-safe user, item, and user-item features for candidate pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fashion_recommender.content_based import ITEM_FEATURE_COLUMNS


USER_FEATURES = [
    "user_total_purchases",
    "user_unique_items",
    "user_active_days",
    "user_days_since_last_purchase",
    "user_average_price",
    "user_online_share",
    "user_age",
]

ITEM_FEATURES = [
    "item_total_purchases",
    "item_unique_customers",
    "item_popularity_7d",
    "item_popularity_30d",
    "item_average_price",
    "item_days_since_last_purchase",
]

PAIR_FEATURES = [
    "user_bought_item_before",
    "user_item_purchase_count",
    "days_since_user_bought_item",
    "user_product_type_count",
    "user_colour_count",
    "user_section_count",
    "user_garment_group_count",
]

GENERATOR_FEATURES = [
    "als_score",
    "als_rank",
    "content_similarity_score",
    "content_rank",
    "personal_history_score",
    "personal_history_rank",
    "popularity_score",
    "popularity_rank",
    "from_als",
    "from_content_based",
    "from_personal_history",
    "from_popularity",
    "number_of_candidate_sources",
]

CATEGORY_COUNT_COLUMNS = {
    "product_type_name": "user_product_type_count",
    "colour_group_name": "user_colour_count",
    "section_name": "user_section_count",
    "garment_group_name": "user_garment_group_count",
}


def _validate_feature_inputs(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    articles: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> None:
    if candidates.duplicated(["customer_id", "article_id"]).any():
        raise ValueError("candidates must contain one row per customer_id/article_id")
    history_required = {"customer_id", "article_id", "t_dat"}
    missing = history_required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing required columns: {sorted(missing)}")
    article_required = {"article_id", *ITEM_FEATURE_COLUMNS}
    missing = article_required - set(articles.columns)
    if missing:
        raise ValueError(f"articles is missing required columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(history["t_dat"]):
        raise TypeError("history['t_dat'] must have a datetime dtype")
    if not history.empty and history["t_dat"].max() >= reference_date:
        raise ValueError("history must contain only dates before reference_date")


def build_candidate_features(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame | None = None,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Add a compact, explainable feature set computed strictly from history."""
    if history.empty:
        raise ValueError("history must not be empty")
    reference = (
        pd.Timestamp(history["t_dat"].max()) + pd.Timedelta(days=1)
        if reference_date is None
        else pd.Timestamp(reference_date)
    )
    _validate_feature_inputs(candidates, history, articles, reference)
    result = candidates.copy()
    result["customer_id"] = result["customer_id"].astype(str)
    result["article_id"] = result["article_id"].astype(str)
    history_work = history.copy()
    history_work["customer_id"] = history_work["customer_id"].astype(str)
    history_work["article_id"] = history_work["article_id"].astype(str)

    user_aggregation = {
        "user_total_purchases": ("article_id", "size"),
        "user_unique_items": ("article_id", "nunique"),
        "user_active_days": ("t_dat", "nunique"),
        "user_last_purchase": ("t_dat", "max"),
    }
    if "price" in history_work:
        user_aggregation["user_average_price"] = ("price", "mean")
    user_features = history_work.groupby("customer_id", as_index=False).agg(
        **user_aggregation
    )
    user_features["user_days_since_last_purchase"] = (
        reference - user_features.pop("user_last_purchase")
    ).dt.days
    if "sales_channel_id" in history_work:
        online_share = (
            history_work.assign(_online=history_work["sales_channel_id"].eq(2).astype(float))
            .groupby("customer_id", as_index=False)["_online"]
            .mean()
            .rename(columns={"_online": "user_online_share"})
        )
        user_features = user_features.merge(online_share, on="customer_id", how="left")

    if customers is not None and {"customer_id", "age"} <= set(customers.columns):
        age = customers.loc[:, ["customer_id", "age"]].copy()
        age["customer_id"] = age["customer_id"].astype(str)
        age = age.drop_duplicates("customer_id").rename(columns={"age": "user_age"})
        user_features = user_features.merge(age, on="customer_id", how="left")
    result = result.merge(user_features, on="customer_id", how="left")

    item_aggregation = {
        "item_total_purchases": ("customer_id", "size"),
        "item_unique_customers": ("customer_id", "nunique"),
        "item_last_purchase": ("t_dat", "max"),
    }
    if "price" in history_work:
        item_aggregation["item_average_price"] = ("price", "mean")
    item_features = history_work.groupby("article_id", as_index=False).agg(
        **item_aggregation
    )
    item_features["item_days_since_last_purchase"] = (
        reference - item_features.pop("item_last_purchase")
    ).dt.days
    for days in (7, 30):
        recent_counts = (
            history_work.loc[
                history_work["t_dat"] >= reference - pd.Timedelta(days=days)
            ]
            .groupby("article_id", as_index=False)
            .size()
            .rename(columns={"size": f"item_popularity_{days}d"})
        )
        item_features = item_features.merge(recent_counts, on="article_id", how="left")
    result = result.merge(item_features, on="article_id", how="left")

    article_attributes = articles.loc[:, ["article_id", *ITEM_FEATURE_COLUMNS]].copy()
    article_attributes["article_id"] = article_attributes["article_id"].astype(str)
    article_attributes = article_attributes.drop_duplicates("article_id")
    article_attributes[ITEM_FEATURE_COLUMNS] = (
        article_attributes[ITEM_FEATURE_COLUMNS].fillna("Unknown").astype(str)
    )
    result = result.merge(article_attributes, on="article_id", how="left")

    pair_features = (
        history_work.groupby(["customer_id", "article_id"], as_index=False)
        .agg(
            user_item_purchase_count=("article_id", "size"),
            user_item_last_purchase=("t_dat", "max"),
        )
    )
    pair_features["days_since_user_bought_item"] = (
        reference - pair_features.pop("user_item_last_purchase")
    ).dt.days
    pair_features["user_bought_item_before"] = 1
    result = result.merge(pair_features, on=["customer_id", "article_id"], how="left")

    history_categories = history_work[["customer_id", "article_id"]].merge(
        article_attributes,
        on="article_id",
        how="left",
    )
    for category, output_column in CATEGORY_COUNT_COLUMNS.items():
        counts = (
            history_categories.groupby(["customer_id", category], as_index=False)
            .size()
            .rename(columns={"size": output_column})
        )
        result = result.merge(counts, on=["customer_id", category], how="left")

    for column in ITEM_FEATURE_COLUMNS:
        result[column] = result[column].fillna("Unknown").astype(str)
    for column in USER_FEATURES + ITEM_FEATURES + PAIR_FEATURES + GENERATOR_FEATURES:
        if column not in result:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)

    validate_feature_table(result)
    return result


def validate_feature_table(feature_table: pd.DataFrame) -> None:
    if feature_table.duplicated(["customer_id", "article_id"]).any():
        raise ValueError("feature table contains duplicate user-item pairs")
    numeric = feature_table.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("feature table contains infinite numeric values")
    if feature_table.isna().any().any():
        missing = feature_table.columns[feature_table.isna().any()].tolist()
        raise ValueError(f"feature table contains missing values: {missing}")


__all__ = [
    "GENERATOR_FEATURES",
    "ITEM_FEATURES",
    "PAIR_FEATURES",
    "USER_FEATURES",
    "build_candidate_features",
    "validate_feature_table",
]
