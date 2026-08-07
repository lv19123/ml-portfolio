import ast
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def notebook_code(name: str) -> str:
    notebook = json.loads((PROJECT_ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def load_notebook(name: str) -> dict:
    path = PROJECT_ROOT / "notebooks" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_notebook_artifact_chain_and_no_retraining() -> None:
    als = notebook_code("05_als_colab.ipynb")
    content = notebook_code("06_content_based_colab.ipynb")
    candidates = notebook_code("07_candidate_generation_colab.ipynb")
    features = notebook_code("08_feature_engineering_colab.ipynb")
    ranking = notebook_code("09_catboost_ranking_colab.ipynb")

    for split in ("train", "validation", "test"):
        assert f"als_candidates_{split}.parquet" in als
        assert f"content_candidates_{split}.parquet" in content
        assert f"als_candidates_{split}.parquet" in candidates
        assert f"content_candidates_{split}.parquet" in candidates
        assert f"merged_candidates_{split}.parquet" in candidates
        assert f"{split}_ranking_table.parquet" in features
    assert "merged_candidates_{split}.parquet" in features
    assert 'CANDIDATE_PATHS["validation"]' in features
    for artifact in (
        "train_ranking_table.parquet",
        "validation_ranking_table.parquet",
        "test_ranking_table.parquet",
    ):
        assert artifact in ranking

    assert "AlternatingLeastSquares" not in candidates + features + ranking
    assert "OneHotEncoder" not in candidates + features + ranking
    assert "fit_transform" not in candidates + features + ranking
    assert "build_candidate_features" not in features + ranking
    assert "prepare_window_table" not in ranking


def test_every_code_cell_is_small_and_has_markdown_context() -> None:
    for path in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            assert len(source.splitlines()) <= 25, f"{path.name}: cell {index}"
            assert index > 0
            assert notebook["cells"][index - 1]["cell_type"] == "markdown"


def test_no_temporal_window_loop_contains_ml_pipeline() -> None:
    forbidden_inside_window_loop = (
        ".fit(",
        "predict_proba",
        "to_parquet",
        "AlternatingLeastSquares",
        "OneHotEncoder",
    )
    for path in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
        notebook = load_notebook(path.name)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            for node in ast.walk(tree):
                if not isinstance(node, ast.For):
                    continue
                loop_text = ast.unparse(node)
                mentions_windows = all(
                    split in loop_text for split in ("train", "validation", "test")
                )
                if mentions_windows:
                    assert not any(
                        forbidden in loop_text
                        for forbidden in forbidden_inside_window_loop
                    )


def test_main_parquet_artifact_schemas_without_hm_dataset(tmp_path: Path) -> None:
    artifacts = {
        "als_candidates.parquet": pd.DataFrame(
            {"customer_id": ["u1"], "article_id": ["0000000001"], "als_score": [0.8], "als_rank": [1]}
        ),
        "content_candidates.parquet": pd.DataFrame(
            {"customer_id": ["u1"], "article_id": ["0000000001"], "content_similarity_score": [0.7], "content_rank": [1]}
        ),
        "merged_candidates.parquet": pd.DataFrame(
            {"customer_id": ["u1"], "article_id": ["0000000001"], "number_of_candidate_sources": [2]}
        ),
        "train_ranking_table.parquet": pd.DataFrame(
            {"customer_id": ["u1"], "article_id": ["0000000001"], "target": [1]}
        ),
        "final_recommendations.parquet": pd.DataFrame(
            {"customer_id": ["u1"], "article_id": ["0000000001"], "rank": [1], "score": [0.9]}
        ),
    }
    required_columns = {
        "als_candidates.parquet": {"customer_id", "article_id", "als_score", "als_rank"},
        "content_candidates.parquet": {"customer_id", "article_id", "content_similarity_score", "content_rank"},
        "merged_candidates.parquet": {"customer_id", "article_id", "number_of_candidate_sources"},
        "train_ranking_table.parquet": {"customer_id", "article_id", "target"},
        "final_recommendations.parquet": {"customer_id", "article_id", "rank", "score"},
    }
    for filename, frame in artifacts.items():
        path = tmp_path / filename
        frame.to_parquet(path, index=False)
        loaded = pd.read_parquet(path)
        assert required_columns[filename] <= set(loaded.columns)
        assert not loaded.duplicated(["customer_id", "article_id"]).any()
