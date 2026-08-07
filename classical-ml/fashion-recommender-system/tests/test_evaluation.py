import pandas as pd
import pytest

from fashion_recommender.evaluation import (
    average_precision_at_k,
    build_ground_truth,
    build_temporal_windows,
    candidate_recall_at_k,
    hit_rate_at_k,
    map_at_k,
    mean_recall_at_k,
    recall_at_k,
    temporal_split,
)


def transactions_for_split() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t_dat": pd.date_range("2020-01-01", periods=14),
            "customer_id": ["u1"] * 14,
            "article_id": [f"{number:010d}" for number in range(14)],
        }
    )


def test_temporal_split_uses_last_seven_calendar_days_and_preserves_input() -> None:
    transactions = transactions_for_split()
    original = transactions.copy(deep=True)
    history, future, cutoff = temporal_split(transactions, future_days=7)
    assert cutoff == pd.Timestamp("2020-01-08")
    assert history["t_dat"].max() == pd.Timestamp("2020-01-07")
    assert future["t_dat"].min() == pd.Timestamp("2020-01-08")
    assert len(history) + len(future) == len(transactions)
    pd.testing.assert_frame_equal(transactions, original)


def test_temporal_split_validation_errors() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        temporal_split(pd.DataFrame({"t_dat": pd.Series(dtype="datetime64[ns]")}), 7)
    with pytest.raises(ValueError, match="positive"):
        temporal_split(transactions_for_split(), 0)
    bad_dates = transactions_for_split().assign(t_dat=lambda frame: frame.t_dat.astype(str))
    with pytest.raises(TypeError, match="datetime"):
        temporal_split(bad_dates, 7)
    short = transactions_for_split().iloc[:3].assign(t_dat=pd.Timestamp("2020-01-01"))
    with pytest.raises(ValueError, match="empty part"):
        temporal_split(short, 7)


def test_build_temporal_windows_are_ordered_and_non_overlapping() -> None:
    transactions = pd.DataFrame(
        {
            "t_dat": pd.date_range("2020-01-01", periods=40),
            "customer_id": ["u1"] * 40,
            "article_id": ["0000000001"] * 40,
        }
    )
    windows = build_temporal_windows(transactions, n_windows=4)
    assert [window.name for window in windows] == [
        "history_window_1",
        "history_window_2",
        "validation_history",
        "final_history",
    ]
    assert all(window.history.t_dat.max() < window.target.t_dat.min() for window in windows)
    assert windows[-1].target_end_date == pd.Timestamp("2020-02-09")


def test_ground_truth_excludes_cold_start_and_deduplicates_in_order() -> None:
    history = pd.DataFrame({"customer_id": ["u1"], "article_id": ["A"]})
    future = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1", "cold"],
            "article_id": ["B", "B", "C", "D"],
        }
    )
    truth, cold = build_ground_truth(history, future)
    assert truth == {"u1": ["B", "C"]}
    assert cold == ["cold"]


@pytest.mark.parametrize(
    ("recommended", "expected"),
    [
        (["A", "B"], 1.0),
        (["A", "X"], 0.5),
        (["X", "Y"], 0.0),
        ([], 0.0),
        (["A", "A"], 0.5),
    ],
)
def test_recall_at_k(recommended, expected) -> None:
    assert recall_at_k(["A", "B"], recommended, 2) == expected


def test_average_precision_does_not_count_duplicate_hits_twice() -> None:
    assert average_precision_at_k(["A", "B"], ["A", "A", "B"], 3) == 1.0


def test_aggregate_metrics_and_candidate_recall() -> None:
    truth = {"u1": ["A", "B"], "u2": ["C"]}
    recs = {"u1": ["A", "X"], "u2": ["C"]}
    assert mean_recall_at_k(truth, recs, 2) == 0.75
    assert map_at_k(truth, recs, 2) == 0.75
    assert hit_rate_at_k(truth, recs, 2) == 1.0
    assert candidate_recall_at_k(truth, recs, 2) == 0.75
