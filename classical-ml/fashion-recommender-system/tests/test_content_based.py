import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from fashion_recommender.content_based import (
    ITEM_FEATURE_COLUMNS,
    build_user_profiles,
    calculate_frequency_weight,
    calculate_recency_weight,
    fit_content_encoder,
    generate_content_candidates,
)


def article_frame() -> pd.DataFrame:
    rows = []
    for article_id, first, second in [("A", "top", "red"), ("B", "top", "red"), ("C", "trouser", "blue")]:
        row = {"article_id": article_id}
        for index, column in enumerate(ITEM_FEATURE_COLUMNS):
            row[column] = first if index % 2 == 0 else second
        rows.append(row)
    return pd.DataFrame(rows)


def test_encoder_returns_sparse_item_matrix() -> None:
    content = fit_content_encoder(article_frame())
    assert isinstance(content.article_feature_matrix, csr_matrix)
    assert content.article_feature_matrix.shape[0] == 3


def test_recency_and_frequency_weights() -> None:
    weights = calculate_recency_weight([0, 30], decay_days=30)
    assert np.isclose(weights[0], 1.0)
    assert np.isclose(weights[1], np.exp(-1))
    frequency = calculate_frequency_weight([1, 3])
    assert frequency[1] > frequency[0]


def test_weighted_profile_shape_and_similar_item_score() -> None:
    articles = article_frame()
    content = fit_content_encoder(articles)
    history = pd.DataFrame(
        {"customer_id": ["u1"], "article_id": ["A"], "t_dat": pd.to_datetime(["2020-01-01"])}
    )
    profiles = build_user_profiles(history, content, pd.Timestamp("2020-01-02"))
    assert profiles.matrix.shape == (1, content.article_feature_matrix.shape[1])
    result = generate_content_candidates(
        profiles,
        content,
        ["u1"],
        seen_items={"u1": {"A"}},
        limit=2,
    )
    assert result.iloc[0]["article_id"] == "B"
    assert result.iloc[0]["content_similarity_score"] > result.iloc[1]["content_similarity_score"]


def test_unknown_user_receives_fallback() -> None:
    content = fit_content_encoder(article_frame())
    history = pd.DataFrame(
        {"customer_id": ["u1"], "article_id": ["A"], "t_dat": pd.to_datetime(["2020-01-01"])}
    )
    profiles = build_user_profiles(history, content)
    result = generate_content_candidates(
        profiles,
        content,
        ["new"],
        fallback_items=["C", "B"],
        limit=2,
    )
    assert result["article_id"].tolist() == ["C", "B"]
    assert result["content_similarity_score"].tolist() == [0.0, 0.0]
