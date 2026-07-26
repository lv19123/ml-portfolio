import pandas as pd
import pytest

from src.features import safe_divide
from src.validation import (
    merge_feature_table,
    validate_application_data,
    validate_feature_table,
    validate_inference_data,
    validate_required_features,
)


def test_application_checks_key_and_binary_target():
    application = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 1],
        }
    )
    validate_application_data(application)

    invalid_target = application.copy()
    invalid_target.loc[0, "TARGET"] = 2
    with pytest.raises(ValueError, match="0 и 1"):
        validate_application_data(invalid_target)


def test_feature_table_has_one_row_per_client_and_no_target():
    features = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "BUREAU_CREDIT_COUNT": [3, 1],
        }
    )
    validate_feature_table(features, allowed_prefixes=("BUREAU_",))

    duplicated = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="уникальным"):
        validate_feature_table(duplicated)


def test_merge_does_not_multiply_application_rows():
    application = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 0],
        }
    )
    features = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 3],
            "PREV_APPLICATION_COUNT": [2, 4],
        }
    )

    merged = merge_feature_table(application, features)

    assert len(merged) == len(application)
    assert merged["SK_ID_CURR"].is_unique
    assert "PREV_APPLICATION_COUNT" in merged.columns


def test_safe_divide_replaces_zero_denominator_with_missing_value():
    result = safe_divide(
        pd.Series([4.0, 2.0]),
        pd.Series([2.0, 0.0]),
    )

    assert result.iloc[0] == 2.0
    assert pd.isna(result.iloc[1])


def test_inference_requires_id_and_forbids_target():
    inference = pd.DataFrame(
        {
            "SK_ID_CURR": [10, 11],
            "AMT_CREDIT": [100.0, 200.0],
        }
    )
    validate_inference_data(inference)

    with_target = inference.assign(TARGET=[0, 1])
    with pytest.raises(ValueError, match="не должен присутствовать"):
        validate_inference_data(with_target)

    with pytest.raises(ValueError, match="SK_ID_CURR"):
        validate_inference_data(inference.drop(columns="SK_ID_CURR"))


def test_application_rejects_empty_and_null_client_id():
    with pytest.raises(ValueError, match="не содержит строк"):
        validate_application_data(
            pd.DataFrame(columns=["SK_ID_CURR", "TARGET"])
        )

    invalid = pd.DataFrame(
        {"SK_ID_CURR": [1, None], "TARGET": [0, 1]}
    )
    with pytest.raises(ValueError, match="пропуски"):
        validate_application_data(invalid)


def test_required_feature_schema_has_clear_error():
    data = pd.DataFrame({"known": [1]})
    validate_required_features(data, ["known"])

    with pytest.raises(ValueError, match="missing"):
        validate_required_features(data, ["known", "missing"])
