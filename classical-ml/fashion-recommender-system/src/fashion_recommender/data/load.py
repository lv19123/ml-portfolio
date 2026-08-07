"""Utilities for loading the raw H&M CSV datasets."""

from pathlib import Path

import pandas as pd


def normalize_article_ids(series: pd.Series) -> pd.Series:
    """Return article identifiers in the canonical ten-character format.

    The H&M transaction file is often read with nine-digit numeric identifiers,
    while ``articles.csv`` contains the leading zero.  Keeping this conversion
    in one place prevents silent join failures in later notebooks.
    """
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )


def _validate_file(path: str | Path) -> Path:
    """Return an absolute file path or raise ``FileNotFoundError``."""
    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"Path does not point to a file: {file_path}")

    return file_path


def _validate_required_columns(
    path: Path,
    required_columns: tuple[str, ...],
) -> None:
    """Raise ``ValueError`` when a CSV header lacks required columns."""
    columns = pd.read_csv(path, nrows=0).columns
    missing_columns = [
        column for column in required_columns if column not in columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {path}: {missing_columns}"
        )


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load transactions while preserving IDs and parsing transaction dates.

    Args:
        path: Path to the transactions CSV file.

    Returns:
        A DataFrame with ``t_dat`` parsed as datetime and customer and article
        identifiers stored as strings.

    Raises:
        FileNotFoundError: If the path does not exist or is not a file.
        ValueError: If any required transaction column is missing.
    """
    file_path = _validate_file(path)
    _validate_required_columns(
        file_path,
        ("t_dat", "customer_id", "article_id"),
    )

    transactions = pd.read_csv(
        file_path,
        dtype={"customer_id": "string", "article_id": "string"},
        parse_dates=["t_dat"],
    )
    transactions["customer_id"] = transactions["customer_id"].astype("string")
    transactions["article_id"] = normalize_article_ids(
        transactions["article_id"]
    )
    return transactions


def load_articles(path: str | Path) -> pd.DataFrame:
    """Load articles while preserving ``article_id`` as a string.

    Args:
        path: Path to the articles CSV file.

    Returns:
        The unmodified CSV contents with string article identifiers.

    Raises:
        FileNotFoundError: If the path does not exist or is not a file.
        ValueError: If the ``article_id`` column is missing.
    """
    file_path = _validate_file(path)
    _validate_required_columns(file_path, ("article_id",))

    articles = pd.read_csv(file_path, dtype={"article_id": "string"})
    articles["article_id"] = normalize_article_ids(articles["article_id"])
    return articles


def load_customers(path: str | Path) -> pd.DataFrame:
    """Load customers while preserving ``customer_id`` as a string.

    Args:
        path: Path to the customers CSV file.

    Returns:
        The unmodified CSV contents with string customer identifiers.

    Raises:
        FileNotFoundError: If the path does not exist or is not a file.
        ValueError: If the ``customer_id`` column is missing.
    """
    file_path = _validate_file(path)
    _validate_required_columns(file_path, ("customer_id",))

    customers = pd.read_csv(file_path, dtype={"customer_id": "string"})
    customers["customer_id"] = customers["customer_id"].astype("string")
    return customers
