from pathlib import Path


def test_sample_lecture_exists_and_mentions_linear_regression() -> None:
    sample_path = Path("sample_data/ml_lecture_sample.txt")

    assert sample_path.exists()
    text = sample_path.read_text(encoding="utf-8")
    assert text.strip()
    assert "линейная регрессия" in text.lower()
