import pandas as pd
import pytest

from fashion_recommender.candidates import (
    merge_candidate_sources,
    popularity_candidates,
)


def source_frames():
    als = pd.DataFrame(
        {"customer_id": ["u1", "u1"], "article_id": ["A", "B"], "als_score": [0.9, 0.5], "als_rank": [1, 2]}
    )
    content = pd.DataFrame(
        {"customer_id": ["u1", "u1"], "article_id": ["A", "C"], "content_similarity_score": [0.8, 0.4], "content_rank": [1, 2]}
    )
    return als, content


def test_merge_sources_keeps_one_pair_flags_and_source_count() -> None:
    als, content = source_frames()
    result = merge_candidate_sources(als=als, content_based=content)
    assert len(result) == 3
    row = result.loc[result["article_id"] == "A"].iloc[0]
    assert row["from_als"] == 1
    assert row["from_content_based"] == 1
    assert row["number_of_candidate_sources"] == 2
    assert not result.duplicated(["customer_id", "article_id"]).any()


def test_merge_sources_applies_per_user_limit() -> None:
    als, content = source_frames()
    result = merge_candidate_sources(als=als, content_based=content, limit_per_user=2)
    assert len(result) == 2
    assert result.iloc[0]["article_id"] == "A"


def test_popularity_candidates_has_requested_users() -> None:
    popular = pd.DataFrame(
        {"article_id": ["A", "B"], "popularity_score": [5, 4], "popularity_rank": [1, 2]}
    )
    result = popularity_candidates(["u1", "u2"], popular, limit=1)
    assert result.groupby("customer_id").size().to_dict() == {"u1": 1, "u2": 1}


def test_empty_source_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        merge_candidate_sources()
