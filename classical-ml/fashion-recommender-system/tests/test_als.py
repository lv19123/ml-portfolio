import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from fashion_recommender.als import build_matrix_with_mappings, prepare_user_item_matrix


def test_prepare_user_item_matrix_is_sparse_and_uses_confidence() -> None:
    history = pd.DataFrame(
        {"customer_id": ["u1", "u1", "u1", "u2"], "article_id": ["A", "A", "B", "B"]}
    )
    result = prepare_user_item_matrix(history)
    assert isinstance(result.matrix, csr_matrix)
    assert result.matrix.shape == (2, 2)
    assert np.isclose(result.matrix[result.user_to_index["u1"], result.item_to_index["A"]], 1 + np.log1p(2))


def test_build_matrix_with_mappings_ignores_unknown_ids() -> None:
    history = pd.DataFrame(
        {"customer_id": ["u1", "new"], "article_id": ["A", "new_item"]}
    )
    result = build_matrix_with_mappings(history, ["u1"], ["A"])
    assert result.matrix.shape == (1, 1)
    assert result.matrix.nnz == 1
