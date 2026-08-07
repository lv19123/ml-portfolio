import pandas as pd
import pytest

from fashion_recommender.ranking import (
    build_top_k_recommendations,
)


def test_top_k_removes_duplicates_and_uses_popularity_fallback() -> None:
    scored = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "article_id": ["A", "A", "B"],
            "score": [0.9, 0.8, 0.7],
        }
    )
    result = build_top_k_recommendations(
        scored,
        ["B", "C", "D"],
        k=3,
        customer_ids=["u1", "new"],
    )
    assert result.loc[result.customer_id == "u1", "article_id"].tolist() == ["A", "B", "C"]
    assert result.loc[result.customer_id == "new", "article_id"].tolist() == ["B", "C", "D"]


def test_top_k_validates_k() -> None:
    scored = pd.DataFrame(columns=["customer_id", "article_id", "score"])
    with pytest.raises(ValueError, match="between 1 and 50"):
        build_top_k_recommendations(scored, [], k=51)
