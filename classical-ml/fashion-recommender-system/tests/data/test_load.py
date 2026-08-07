"""Tests for raw CSV loading utilities."""

from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from fashion_recommender.data import (
    load_articles,
    load_customers,
    load_transactions,
)


@pytest.fixture
def transactions_csv(tmp_path: Path) -> Path:
    """Create a minimal transactions CSV file."""
    csv_path = tmp_path / "transactions.csv"
    csv_path.write_text(
        "t_dat,customer_id,article_id,price\n"
        "2020-09-01,customer-001,0012345678,0.05\n"
        "2020-09-02,customer-002,0000000042,0.10\n",
        encoding="utf-8",
    )
    return csv_path


def test_load_transactions_success(transactions_csv: Path) -> None:
    result = load_transactions(transactions_csv)

    assert result.shape == (2, 4)
    assert result["price"].tolist() == [0.05, 0.10]


def test_load_transactions_parses_date(transactions_csv: Path) -> None:
    result = load_transactions(transactions_csv)

    assert pd.api.types.is_datetime64_ns_dtype(result["t_dat"])
    assert result.loc[0, "t_dat"] == pd.Timestamp("2020-09-01")


def test_load_transactions_preserves_customer_id_as_string(
    transactions_csv: Path,
) -> None:
    result = load_transactions(transactions_csv)

    assert isinstance(result.loc[0, "customer_id"], str)
    assert pd.api.types.is_string_dtype(result["customer_id"])


def test_load_transactions_preserves_article_id_as_string(
    transactions_csv: Path,
) -> None:
    result = load_transactions(transactions_csv)

    assert isinstance(result.loc[0, "article_id"], str)
    assert pd.api.types.is_string_dtype(result["article_id"])


def test_load_transactions_preserves_article_id_leading_zeroes(
    transactions_csv: Path,
) -> None:
    result = load_transactions(transactions_csv)

    assert result["article_id"].tolist() == ["0012345678", "0000000042"]


def test_load_transactions_normalizes_numeric_like_article_ids(tmp_path: Path) -> None:
    csv_path = tmp_path / "numeric_ids.csv"
    csv_path.write_text(
        "t_dat,customer_id,article_id\n2020-01-01,u1,123.0\n",
        encoding="utf-8",
    )
    assert load_transactions(csv_path).loc[0, "article_id"] == "0000000123"


def test_load_articles_success(tmp_path: Path) -> None:
    csv_path = tmp_path / "articles.csv"
    csv_path.write_text(
        "article_id,prod_name\n0000000042,T-shirt\n",
        encoding="utf-8",
    )

    result = load_articles(csv_path)

    assert result.loc[0, "article_id"] == "0000000042"
    assert result.loc[0, "prod_name"] == "T-shirt"


def test_load_customers_success(tmp_path: Path) -> None:
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "customer_id,age\n000customer,25\n",
        encoding="utf-8",
    )

    result = load_customers(csv_path)

    assert result.loc[0, "customer_id"] == "000customer"
    assert result.loc[0, "age"] == 25


def test_missing_file_raises_with_absolute_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match=str(missing_path.resolve())):
        load_articles(missing_path)


def test_directory_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=str(tmp_path.resolve())):
        load_customers(tmp_path)


@pytest.mark.parametrize(
    ("loader", "csv_content", "missing_columns"),
    [
        (
            load_transactions,
            "price,sales_channel_id\n0.05,1\n",
            ["t_dat", "customer_id", "article_id"],
        ),
        (load_articles, "prod_name\nT-shirt\n", ["article_id"]),
        (load_customers, "age\n25\n", ["customer_id"]),
    ],
)
def test_missing_required_columns_raise_value_error(
    tmp_path: Path,
    loader: Callable[[str | Path], pd.DataFrame],
    csv_content: str,
    missing_columns: list[str],
) -> None:
    csv_path = tmp_path / "missing_columns.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    with pytest.raises(ValueError) as error:
        loader(csv_path)

    assert str(missing_columns) in str(error.value)
