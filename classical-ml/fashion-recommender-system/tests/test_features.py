import pandas as pd
import pytest

from fashion_recommender.content_based import ITEM_FEATURE_COLUMNS
from fashion_recommender.features import build_candidate_features


def inputs():
    candidates = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u2"],
            "article_id": ["A", "B", "C"],
            "als_score": [0.9, 0.2, 0.1],
            "als_rank": [1, 2, 1],
        }
    )
    history = pd.DataFrame(
        {
            "t_dat": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "customer_id": ["u1", "u1", "u2"],
            "article_id": ["A", "A", "C"],
            "price": [0.1, 0.2, 0.3],
            "sales_channel_id": [2, 1, 2],
        }
    )
    articles = []
    for article_id, category in [("A", "top"), ("B", "top"), ("C", "trouser")]:
        row = {"article_id": article_id}
        row.update({column: category for column in ITEM_FEATURE_COLUMNS})
        articles.append(row)
    customers = pd.DataFrame({"customer_id": ["u1", "u2"], "age": [25, None]})
    return candidates, history, pd.DataFrame(articles), customers


def test_build_candidate_features_values_no_missing_and_no_input_mutation() -> None:
    candidates, history, articles, customers = inputs()
    original_history = history.copy(deep=True)
    result = build_candidate_features(
        candidates,
        history,
        articles,
        customers,
        reference_date=pd.Timestamp("2020-01-03"),
    )
    u1_a = result.loc[(result.customer_id == "u1") & (result.article_id == "A")].iloc[0]
    assert u1_a["user_total_purchases"] == 2
    assert u1_a["user_item_purchase_count"] == 2
    assert u1_a["user_bought_item_before"] == 1
    assert u1_a["user_product_type_count"] == 2
    assert not result.isna().any().any()
    pd.testing.assert_frame_equal(history, original_history)


def test_features_reject_history_at_or_after_cutoff() -> None:
    candidates, history, articles, customers = inputs()
    with pytest.raises(ValueError, match="before reference_date"):
        build_candidate_features(
            candidates,
            history,
            articles,
            customers,
            reference_date=pd.Timestamp("2020-01-02"),
        )
