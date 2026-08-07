import pandas as pd

from fashion_recommender.baselines import (
    personal_history_candidates,
    popular_items,
)


def history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t_dat": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
            ),
            "customer_id": ["u1", "u1", "u1", "u2", "u2"],
            "article_id": ["A", "B", "A", "B", "C"],
        }
    )


def test_popularity_counts_history_only() -> None:
    result = popular_items(history(), 3)
    assert result["article_id"].tolist()[:2] == ["A", "B"]
    assert result["popularity_score"].tolist()[:2] == [2, 2]


def test_popularity_ties_are_independent_of_input_order() -> None:
    original = history()
    expected = popular_items(original, 3)

    for random_state in (1, 7, 42):
        shuffled = original.sample(frac=1, random_state=random_state)
        actual = popular_items(shuffled, 3)
        pd.testing.assert_frame_equal(actual, expected)


def test_personal_candidates_have_ranks_and_limit() -> None:
    result = personal_history_candidates(history(), limit=1)
    assert result.groupby("customer_id").size().max() == 1
    assert set(result.columns) == {
        "customer_id",
        "article_id",
        "personal_history_score",
        "personal_history_rank",
    }
