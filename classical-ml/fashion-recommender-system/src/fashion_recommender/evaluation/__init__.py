"""Temporal validation and ranking metrics for recommender experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalWindow:
    """One expanding-history window and its following target period."""

    name: str
    history: pd.DataFrame
    target: pd.DataFrame
    cutoff_date: pd.Timestamp
    target_end_date: pd.Timestamp


def _validate_temporal_input(
    transactions: pd.DataFrame,
    future_days: int,
) -> None:
    if "t_dat" not in transactions.columns:
        raise ValueError("transactions must contain the 't_dat' column")
    if not pd.api.types.is_datetime64_any_dtype(transactions["t_dat"]):
        raise TypeError("transactions['t_dat'] must have a datetime dtype")
    if not isinstance(future_days, int) or isinstance(future_days, bool):
        raise TypeError("future_days must be an integer")
    if future_days <= 0:
        raise ValueError("future_days must be positive")
    if transactions.empty:
        raise ValueError("transactions must not be empty")
    if transactions["t_dat"].isna().any():
        raise ValueError("transactions['t_dat'] must not contain missing dates")


def temporal_split(
    transactions: pd.DataFrame,
    future_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split transactions into expanding history and the last target period."""
    _validate_temporal_input(transactions, future_days)

    last_date = transactions["t_dat"].max()
    cutoff_date = last_date - pd.Timedelta(days=future_days - 1)
    history = transactions.loc[transactions["t_dat"] < cutoff_date].copy()
    future = transactions.loc[transactions["t_dat"] >= cutoff_date].copy()

    if history.empty or future.empty:
        raise ValueError(
            "temporal split produced an empty part; provide a longer date range"
        )
    if history["t_dat"].max() >= future["t_dat"].min():
        raise RuntimeError("history and future overlap in time")
    if len(history) + len(future) != len(transactions):
        raise RuntimeError("temporal split lost transaction rows")

    return history, future, pd.Timestamp(cutoff_date)


def build_temporal_windows(
    transactions: pd.DataFrame,
    target_days: int = 7,
    n_windows: int = 4,
    step_days: int = 7,
    names: Sequence[str] | None = None,
) -> list[TemporalWindow]:
    """Build chronological expanding-history windows without target overlap.

    Windows are returned from oldest to newest.  By convention the last one is
    kept for final evaluation and must not be used to tune model parameters.
    """
    _validate_temporal_input(transactions, target_days)
    if n_windows <= 0:
        raise ValueError("n_windows must be positive")
    if step_days < target_days:
        raise ValueError("step_days must be at least target_days")

    if names is None:
        default_names = [
            "history_window_1",
            "history_window_2",
            "validation_history",
            "final_history",
        ]
        names = (
            default_names[-n_windows:]
            if n_windows <= len(default_names)
            else [f"window_{number}" for number in range(1, n_windows + 1)]
        )
    if len(names) != n_windows:
        raise ValueError("names must contain exactly n_windows values")

    last_date = pd.Timestamp(transactions["t_dat"].max())
    windows: list[TemporalWindow] = []
    for offset, name in reversed(list(enumerate(names))):
        distance_from_last = (n_windows - 1 - offset) * step_days
        target_end = last_date - pd.Timedelta(days=distance_from_last)
        cutoff = target_end - pd.Timedelta(days=target_days - 1)
        history = transactions.loc[transactions["t_dat"] < cutoff].copy()
        target = transactions.loc[
            (transactions["t_dat"] >= cutoff)
            & (transactions["t_dat"] <= target_end)
        ].copy()
        if history.empty or target.empty:
            raise ValueError(
                f"window '{name}' is empty; reduce n_windows or use more data"
            )
        windows.append(
            TemporalWindow(
                name=str(name),
                history=history,
                target=target,
                cutoff_date=pd.Timestamp(cutoff),
                target_end_date=pd.Timestamp(target_end),
            )
        )

    return sorted(windows, key=lambda window: window.cutoff_date)


def build_ground_truth(
    history: pd.DataFrame,
    future: pd.DataFrame,
) -> tuple[dict[str, list[str]], list[str]]:
    """Create ordered unique future purchases for users with known history."""
    required = {"customer_id", "article_id"}
    for frame_name, frame in (("history", history), ("future", future)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{frame_name} is missing required columns: {sorted(missing)}"
            )

    history_users = set(history["customer_id"].astype(str))
    future_users = set(future["customer_id"].astype(str))
    cold_start_users = sorted(future_users - history_users)
    known_future = future.loc[
        future["customer_id"].astype(str).isin(history_users),
        ["customer_id", "article_id"],
    ].copy()
    known_future["customer_id"] = known_future["customer_id"].astype(str)
    known_future["article_id"] = known_future["article_id"].astype(str)

    ground_truth = (
        known_future.groupby("customer_id", sort=False)["article_id"]
        .agg(lambda values: list(dict.fromkeys(values)))
        .to_dict()
    )
    return ground_truth, cold_start_users


def _unique_top_k(items: Iterable[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    return list(dict.fromkeys(str(item) for item in items))[:k]


def recall_at_k(
    actual_items: Iterable[str],
    recommended_items: Iterable[str],
    k: int = 12,
) -> float:
    actual = set(str(item) for item in actual_items)
    if not actual:
        return 0.0
    recommended = set(_unique_top_k(recommended_items, k))
    return len(actual & recommended) / len(actual)


def average_precision_at_k(
    actual_items: Iterable[str],
    recommended_items: Iterable[str],
    k: int = 12,
) -> float:
    actual = set(str(item) for item in actual_items)
    if not actual:
        return 0.0
    score = 0.0
    hits = 0
    for position, item in enumerate(_unique_top_k(recommended_items, k), 1):
        if item in actual:
            hits += 1
            score += hits / position
    return score / min(len(actual), k)


def _mean_metric(
    ground_truth: Mapping[str, Iterable[str]],
    recommendations: Mapping[str, Iterable[str]],
    metric,
    k: int,
) -> float:
    if not ground_truth:
        return 0.0
    scores = [
        metric(actual, recommendations.get(customer_id, []), k)
        for customer_id, actual in ground_truth.items()
    ]
    return float(np.mean(scores))


def mean_recall_at_k(
    ground_truth: Mapping[str, Iterable[str]],
    recommendations: Mapping[str, Iterable[str]],
    k: int = 12,
) -> float:
    return _mean_metric(ground_truth, recommendations, recall_at_k, k)


def map_at_k(
    ground_truth: Mapping[str, Iterable[str]],
    recommendations: Mapping[str, Iterable[str]],
    k: int = 12,
) -> float:
    return _mean_metric(
        ground_truth,
        recommendations,
        average_precision_at_k,
        k,
    )


def hit_rate_at_k(
    ground_truth: Mapping[str, Iterable[str]],
    recommendations: Mapping[str, Iterable[str]],
    k: int = 12,
) -> float:
    if not ground_truth:
        return 0.0
    hits = [
        float(
            bool(
                set(str(item) for item in actual)
                & set(_unique_top_k(recommendations.get(customer_id, []), k))
            )
        )
        for customer_id, actual in ground_truth.items()
    ]
    return float(np.mean(hits))


def candidate_recall_at_k(
    ground_truth: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
    k: int = 100,
) -> float:
    """Return mean recall of true items in each user's candidate set."""
    return mean_recall_at_k(ground_truth, candidates, k)


__all__ = [
    "TemporalWindow",
    "average_precision_at_k",
    "build_ground_truth",
    "build_temporal_windows",
    "candidate_recall_at_k",
    "hit_rate_at_k",
    "map_at_k",
    "mean_recall_at_k",
    "recall_at_k",
    "temporal_split",
]
