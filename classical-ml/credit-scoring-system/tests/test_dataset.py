import pandas as pd
import pytest

from src.dataset import (
    assemble_modeling_data,
    create_or_load_client_split,
    read_application_csv,
    split_modeling_data,
)
from src.features import build_pos_cash_features


def application_frame():
    return pd.DataFrame(
        {
            "SK_ID_CURR": list(range(1, 21)),
            "TARGET": [0, 1] * 10,
            "AMT_CREDIT": [100.0 + value for value in range(20)],
        }
    )


def test_client_split_is_saved_reused_and_stratified(tmp_path):
    application = application_frame()
    path = tmp_path / "processed" / "client_split.csv"

    first = create_or_load_client_split(
        application,
        path=path,
        holdout_size=0.20,
    )
    second = create_or_load_client_split(application, path=path)
    train, holdout = split_modeling_data(application, first)

    pd.testing.assert_frame_equal(first, second)
    assert len(train) == 16
    assert len(holdout) == 4
    assert holdout["TARGET"].mean() == pytest.approx(0.5)


def test_application_loader_handles_target_contract_and_missing_file(tmp_path):
    inference_path = tmp_path / "application_test.csv"
    pd.DataFrame(
        {"SK_ID_CURR": [1], "AMT_CREDIT": [100.0]}
    ).to_csv(inference_path, index=False)

    loaded = read_application_csv(inference_path, require_target=False)

    assert loaded.columns.tolist() == ["SK_ID_CURR", "AMT_CREDIT"]
    with pytest.raises(ValueError, match="TARGET"):
        read_application_csv(inference_path, require_target=True)
    with pytest.raises(FileNotFoundError, match="не найден"):
        read_application_csv(tmp_path / "missing.csv", require_target=False)


def test_feature_merge_preserves_application_rows():
    application = application_frame()
    pos_features = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 3],
            "POS_RECORD_COUNT": [2, 1],
        }
    )

    merged = assemble_modeling_data(
        application,
        feature_set="pos_cash",
        feature_tables={"pos_cash": pos_features},
    )

    assert len(merged) == len(application)
    assert merged["SK_ID_CURR"].is_unique


def test_pos_cash_builder_excludes_future_rows():
    raw = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1],
            "SK_ID_PREV": [10, 10, 11],
            "MONTHS_BALANCE": [-2, -1, 1],
            "SK_DPD": [0, 3, 100],
            "NAME_CONTRACT_STATUS": ["Active", "Active", "Active"],
            "CNT_INSTALMENT_FUTURE": [3.0, 2.0, 1.0],
        }
    )

    features = build_pos_cash_features(raw)

    assert features.loc[0, "POS_RECORD_COUNT"] == 2
    assert features.loc[0, "POS_DPD_MAX"] == 3
