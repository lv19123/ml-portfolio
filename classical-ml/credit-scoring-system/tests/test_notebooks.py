import ast
import json
import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((PROJECT_ROOT / "notebooks").rglob("*.ipynb"))


def read_notebook(name):
    path = PROJECT_ROOT / "notebooks" / name
    return json.loads(path.read_text(encoding="utf-8"))


def code_sources(notebook):
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_expected_notebooks_exist():
    assert [path.name[:2] for path in NOTEBOOKS] == [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
    ]


def test_notebook_code_is_valid_and_saved_outputs_have_no_errors():
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cell_ids = [
            cell.get("id")
            for cell in notebook["cells"]
        ]
        assert len(cell_ids) == len(set(cell_ids)), path.name
        markdown = "\n".join(
            "".join(cell.get("source", []))
            if isinstance(cell.get("source", []), list)
            else cell.get("source", "")
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        assert "## Цель" in markdown, path.name
        assert "## Выводы" in markdown, path.name

        for cell in notebook["cells"]:
            assert cell.get("id"), path.name
            if cell["cell_type"] != "code":
                continue
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            ast.parse(source, filename=f"{path.name}:{cell.get('id', '')}")
            for output in cell.get("outputs", []):
                assert output.get("output_type") != "error", path.name
            execution_count = cell.get("execution_count")
            assert (
                execution_count is None
                or isinstance(execution_count, int)
            ), path.name


def test_notebooks_use_shared_colab_bootstrap_and_config():
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        sources = code_sources(notebook)
        first_code = sources[0]
        setup_and_imports = "\n".join(sources[:2])
        full_text = path.read_text(encoding="utf-8")

        assert '"google.colab" in sys.modules' in first_code
        assert "from src.notebook_setup import setup_notebook" in first_code
        assert "PROJECT_ROOT = setup_notebook()" in first_code
        assert "from src.config import" in setup_and_imports
        assert "find_data_file(" in setup_and_imports
        assert 'PROJECT_ROOT / "data"' not in setup_and_imports
        assert 'PROJECT_ROOT / "models"' not in setup_and_imports
        assert 'PROJECT_ROOT / "reports"' not in setup_and_imports
        assert "sys.path.append(\"../\")" not in full_text
        assert "/Users/" not in full_text
        assert "project_results.xlsx" not in full_text
        assert "openpyxl" not in full_text


def test_eda_does_not_build_service_inventory_tables():
    notebook = read_notebook("01_data_overview.ipynb")
    full_text = "\n".join(code_sources(notebook))

    assert "inventory" not in full_text
    assert "count_csv_rows" not in full_text
    assert "application.info()" in full_text
    assert "value_counts(normalize=True)" in full_text
    assert "application.isna()" in full_text


def test_baseline_shows_sklearn_stages_in_separate_cells():
    notebook = read_notebook("02_application_baseline.ipynb")
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    sources = code_sources(notebook)

    assert "## 5. Preprocessing для Logistic Regression" in markdown
    assert "## 6. Модель 1. Logistic Regression" in markdown
    assert "## 7. Модель 2. CatBoost" in markdown
    assert "## 8. Единый реестр CV-метрик" in markdown

    assert any(text.strip().startswith("numeric_pipeline =") for text in sources)
    assert any(
        text.strip().startswith("categorical_pipeline =")
        for text in sources
    )
    assert any(text.strip().startswith("preprocessor =") for text in sources)
    assert any(text.strip().startswith("logistic_model =") for text in sources)
    assert any(text.strip().startswith("logistic_pipeline =") for text in sources)
    assert any(
        text.strip().startswith("logistic_pipeline.fit(")
        for text in sources
    )

    full_text = "\n".join(sources)
    assert "SimpleImputer(strategy=\"median\")" in full_text
    assert "strategy=\"most_frequent\"" in full_text
    assert "OneHotEncoder(" in full_text
    assert "handle_unknown=\"ignore\"" in full_text
    assert "make_column_selector(" in full_text
    assert "dtype_include=np.number" in full_text
    assert "dtype_exclude=np.number" in full_text
    assert "StratifiedKFold(" in full_text
    assert "cross_validate(" in full_text
    assert "cv=cv_splitter" in full_text
    assert "cross_val_predict(" not in full_text
    assert "preprocessor.fit_transform" not in full_text


def test_feature_notebooks_show_direct_aggregation_and_unique_key_check():
    for name in [
        "03_bureau_features.ipynb",
        "04_previous_application_features.ipynb",
        "05_installments_features.ipynb",
        "06_pos_cash_features.ipynb",
        "07_credit_card_features.ipynb",
    ]:
        notebook = read_notebook(name)
        full_text = "\n".join(code_sources(notebook))

        assert '.groupby("SK_ID_CURR")' in full_text
        assert '["SK_ID_CURR"].is_unique' in full_text
        assert "TARGET" not in "\n".join(
            source
            for source in code_sources(notebook)
            if ".groupby(" in source
        )


def test_final_notebook_uses_explicit_one_to_one_merges():
    notebook = read_notebook("08_final_model.ipynb")
    full_text = "\n".join(code_sources(notebook))

    assert full_text.count("full_data = full_data.merge(") == 5
    assert full_text.count('validate="one_to_one"') >= 6
    assert full_text.count("assert len(full_data) == len(application)") == 5
    assert "merge_feature_table" not in full_text
    assert "for feature_path in" not in full_text
    assert "final_catboost_cv.csv" not in full_text


def test_baseline_creates_and_other_notebooks_read_client_split():
    split_creator = "\n".join(
        code_sources(read_notebook("02_application_baseline.ipynb"))
    )
    assert "train_test_split(" in split_creator
    assert "test_size=0.20" in split_creator
    assert 'stratify=application["TARGET"]' in split_creator
    assert "random_state=RANDOM_STATE" in split_creator
    assert "client_split.to_csv(" in split_creator
    assert 'CLIENT_SPLIT_PATH.exists()' in split_creator
    assert '{"train", "holdout"}' in split_creator

    for name in [
        "03_bureau_features.ipynb",
        "04_previous_application_features.ipynb",
        "05_installments_features.ipynb",
        "06_pos_cash_features.ipynb",
        "07_credit_card_features.ipynb",
        "08_final_model.ipynb",
    ]:
        notebook = read_notebook(name)
        full_text = "\n".join(code_sources(notebook))

        assert 'DATA_PROCESSED_DIR / "client_split.csv"' in full_text
        assert "client_split = pd.read_csv(" in full_text
        assert 'set(client_split["split"]) == {"train", "holdout"}' in full_text
        assert '"split" not in X_train.columns' in full_text
        assert "client_split.to_csv(" not in full_text
        assert "notebooks/02_application_baseline.ipynb" in full_text
        if name == "08_final_model.ipynb":
            assert full_text.count("train_test_split(") == 2
            assert "train_size=50_000" in full_text
            assert "train_size=15_000" in full_text
        else:
            assert "train_test_split(" not in full_text


def test_holdout_predictions_and_metrics_exist_only_in_final_notebook():
    for name in [
        "02_application_baseline.ipynb",
        "03_bureau_features.ipynb",
        "04_previous_application_features.ipynb",
        "05_installments_features.ipynb",
        "06_pos_cash_features.ipynb",
        "07_credit_card_features.ipynb",
    ]:
        full_text = "\n".join(code_sources(read_notebook(name)))
        assert "predict_proba(" not in full_text, name
        assert "roc_auc_score(" not in full_text, name
        assert "average_precision_score(" not in full_text, name
        assert '"holdout_roc_auc": None' in full_text, name
        assert '"holdout_pr_auc": None' in full_text, name

    sources = code_sources(read_notebook("08_final_model.ipynb"))
    fit_cells = [
        text for text in sources
        if text.strip().startswith("catboost_model.fit(")
    ]
    probability_cells = [
        text for text in sources
        if text.strip().startswith(
            "holdout_proba = catboost_model.predict_proba("
        )
    ]
    metric_cells = [
        text for text in sources
        if text.strip().startswith(
            "holdout_roc_auc = roc_auc_score("
        )
    ]

    assert len(fit_cells) == 1
    assert len(probability_cells) == 1
    assert len(metric_cells) == 1
    assert "predict_proba" not in fit_cells[0]
    assert "roc_auc_score" not in probability_cells[0]


def test_all_catboost_notebooks_use_shared_device_configuration():
    for name in [
        "02_application_baseline.ipynb",
        "03_bureau_features.ipynb",
        "04_previous_application_features.ipynb",
        "05_installments_features.ipynb",
        "06_pos_cash_features.ipynb",
        "07_credit_card_features.ipynb",
    ]:
        full_text = "\n".join(code_sources(read_notebook(name)))
        assert "get_catboost_gpu_count()" in full_text, name
        assert "get_catboost_device_config(" in full_text, name
        assert "print_catboost_device_info(" in full_text, name
        assert full_text.count("**catboost_device_config") == 2, name
        assert 'task_type="GPU"' not in full_text, name
        assert 'devices="0"' not in full_text, name

    final_text = "\n".join(
        code_sources(read_notebook("08_final_model.ipynb"))
    )
    assert "get_catboost_gpu_count()" in final_text
    assert "get_catboost_device_config(" not in final_text
    assert "print_catboost_device_info(" not in final_text
    assert final_text.count("**catboost_device_config") == 2
    assert '"task_type": "GPU"' in final_text
    assert '"devices": "0"' in final_text
    assert '"task_type": "CPU"' in final_text
    assert '"thread_count": -1' in final_text


def test_final_notebook_run_modes_and_auto_fallback():
    notebook = read_notebook("08_final_model.ipynb")
    config_source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["id"] == "catboost-device"
    )

    def run_config(requested_mode, gpu_count):
        source = config_source.replace(
            'REQUESTED_RUN_MODE = "auto"',
            f'REQUESTED_RUN_MODE = "{requested_mode}"',
            1,
        )
        namespace = {
            "IN_COLAB": True,
            "os": os,
            "get_catboost_gpu_count": lambda: gpu_count,
        }
        exec(source, namespace)
        return namespace

    auto_cpu = run_config("auto", 0)
    assert auto_cpu["ACTUAL_RUN_MODE"] == "cpu_debug"
    assert auto_cpu["IS_DEBUG"] is True
    assert auto_cpu["CV_FOLDS"] == 2
    assert auto_cpu["MAX_ITERATIONS"] == 150
    assert auto_cpu["EARLY_STOPPING_ROUNDS"] == 30
    assert auto_cpu["catboost_device_config"] == {
        "task_type": "CPU",
        "thread_count": -1,
    }

    auto_gpu = run_config("auto", 1)
    assert auto_gpu["ACTUAL_RUN_MODE"] == "full_gpu"
    assert auto_gpu["IS_DEBUG"] is False
    assert auto_gpu["CV_FOLDS"] == 3
    assert auto_gpu["MAX_ITERATIONS"] == 1000
    assert auto_gpu["EARLY_STOPPING_ROUNDS"] == 100
    assert auto_gpu["catboost_device_config"] == {
        "task_type": "GPU",
        "devices": "0",
    }

    full_cpu = run_config("full_cpu", 0)
    assert full_cpu["ACTUAL_RUN_MODE"] == "full_cpu"
    assert full_cpu["catboost_device_config"]["task_type"] == "CPU"
    assert full_cpu["IS_DEBUG"] is False

    with pytest.raises(RuntimeError, match="full_gpu"):
        run_config("full_gpu", 0)


def test_final_notebook_debug_sampling_and_dynamic_parameters():
    full_text = "\n".join(
        code_sources(read_notebook("08_final_model.ipynb"))
    )

    assert 'REQUESTED_RUN_MODE = "auto"' in full_text
    for mode in [
        "auto",
        "full_gpu",
        "full_cpu",
        "cpu_debug",
    ]:
        assert f'"{mode}"' in full_text

    assert "modeling_data = full_data.merge(" in full_text
    assert "train_size=50_000" in full_text
    assert "train_size=15_000" in full_text
    assert 'stratify=train_data["TARGET"]' in full_text
    assert 'stratify=holdout_data["TARGET"]' in full_text
    assert full_text.count("reset_index(drop=True)") == 2
    assert '"iterations": MAX_ITERATIONS' in full_text
    assert (
        'catboost_params["thread_count"] = CPU_THREAD_COUNT'
        in full_text
    )
    assert (
        "early_stopping_rounds=EARLY_STOPPING_ROUNDS"
        in full_text
    )
    assert "iterations=best_iteration" in full_text


def test_modeling_notebooks_use_builtin_catboost_cv():
    for name in [
        "02_application_baseline.ipynb",
        "03_bureau_features.ipynb",
        "04_previous_application_features.ipynb",
        "05_installments_features.ipynb",
        "06_pos_cash_features.ipynb",
        "07_credit_card_features.ipynb",
        "08_final_model.ipynb",
    ]:
        notebook = read_notebook(name)
        full_text = "\n".join(code_sources(notebook))

        assert "Pool(" in full_text
        assert "cv as catboost_cv" in full_text
        assert "catboost_cv(" in full_text
        assert "folds=cv_splitter" in full_text
        assert "\"custom_metric\": [\"PRAUC:type=Classic\"]" in full_text
        assert "column.startswith(\"test-AUC\")" in full_text
        assert "column.startswith(\"test-PRAUC\")" in full_text
        assert "best_iteration" in full_text
        assert "catboost_cv.csv" not in full_text

        forbidden_fragments = [
            "for fold,",
            "train_idx",
            "valid_idx",
            "train_index",
            "valid_index",
            "oof_proba",
            "full_oof",
            "train_catboost_experiment",
            "train_mask",
            "validation_mask",
            "fold_values",
        ]
        for fragment in forbidden_fragments:
            assert fragment not in full_text


def test_each_feature_experiment_uses_only_its_own_source():
    sources = {
        "03_bureau_features.ipynb": [
            "bureau.csv",
            "bureau_balance.csv",
        ],
        "04_previous_application_features.ipynb": [
            "previous_application.csv",
        ],
        "05_installments_features.ipynb": [
            "installments_payments.csv",
        ],
        "06_pos_cash_features.ipynb": [
            "POS_CASH_balance.csv",
        ],
        "07_credit_card_features.ipynb": [
            "credit_card_balance.csv",
        ],
    }
    all_source_names = {
        source
        for source_list in sources.values()
        for source in source_list
    }

    for name, expected_sources in sources.items():
        full_text = "\n".join(code_sources(read_notebook(name)))
        for expected_source in expected_sources:
            assert expected_source in full_text
        for other_source in all_source_names.difference(expected_sources):
            assert other_source not in full_text


def test_interim_feature_tables_and_common_results_are_saved():
    experiments = {
        "03_bureau_features.ipynb": "application_bureau",
        "04_previous_application_features.ipynb": "application_previous",
        "05_installments_features.ipynb": "application_installments",
        "06_pos_cash_features.ipynb": "application_pos_cash",
        "07_credit_card_features.ipynb": "application_credit_card",
        "08_final_model.ipynb": "application_all_features",
    }

    for name, experiment in experiments.items():
        full_text = "\n".join(code_sources(read_notebook(name)))

        if name == "08_final_model.ipynb":
            assert (
                f'EXPERIMENT = "{experiment}"'
                in full_text
            )
            assert '"experiment": EXPERIMENT' in full_text
        else:
            assert f'"experiment": "{experiment}"' in full_text
        assert "save_experiment_result(current_result)" in full_text
        assert "current_result = {" in full_text
        assert "all_results.to_csv(" not in full_text

        if not name.startswith("08"):
            assert "DATA_INTERIM_DIR" in full_text
            assert "_features.to_csv(" in full_text


def test_baseline_writes_two_models_to_common_registry():
    full_text = "\n".join(
        code_sources(
            read_notebook("02_application_baseline.ipynb")
        )
    )

    assert '"experiment": "application_logistic"' in full_text
    assert '"experiment": "application_catboost"' in full_text
    assert '"model": "LogisticRegression"' in full_text
    assert '"model": "CatBoostClassifier"' in full_text
    assert full_text.count("save_experiment_result(") == 2
    assert '"device": "CPU"' in full_text


def test_only_common_model_metrics_path_is_used():
    project_sources = []
    for path in [
        *PROJECT_ROOT.glob("*.py"),
        *PROJECT_ROOT.glob("src/**/*.py"),
        *PROJECT_ROOT.glob("scripts/**/*.py"),
        *PROJECT_ROOT.glob("tests/**/*.py"),
        *PROJECT_ROOT.glob("notebooks/*.ipynb"),
    ]:
        project_sources.append(path.read_text(encoding="utf-8"))

    full_text = "\n".join(project_sources)
    legacy_names = [
        "baseline_model_" + "comparison.csv",
        "model_" + "comparison.csv",
    ]
    for legacy_name in legacy_names:
        assert legacy_name not in full_text

    tracking_source = (
        PROJECT_ROOT
        / "src"
        / "experiment_tracking.py"
    ).read_text(encoding="utf-8")
    assert '"model_metrics.csv"' in tracking_source


def test_final_notebook_saves_model_and_serializable_metadata():
    full_text = "\n".join(
        code_sources(read_notebook("08_final_model.ipynb"))
    )
    assert 'MODELS_DIR / "catboost_all_features.cbm"' in full_text
    assert (
        'MODELS_DIR / "catboost_all_features_metadata.json"'
        in full_text
    )
    assert (
        'MODELS_DIR / "catboost_all_features_cpu_debug.cbm"'
        in full_text
    )
    assert (
        '"catboost_all_features_cpu_debug_metadata.json"'
        in full_text
    )
    assert "catboost_model.save_model(" in full_text
    assert "json.dump(" in full_text

    metadata_fields = [
        "model_name",
        "experiment",
        "created_at",
        "catboost_version",
        "python_version",
        "run_mode",
        "is_debug",
        "device",
        "gpu_count",
        "random_state",
        "cv_folds",
        "max_iterations",
        "early_stopping_rounds",
        "best_iteration",
        "parameters",
        "feature_names",
        "categorical_features",
        "source_tables",
        "n_train",
        "n_holdout",
        "n_features",
        "cv_roc_auc",
        "cv_roc_auc_std",
        "cv_pr_auc",
        "cv_pr_auc_std",
        "holdout_roc_auc",
        "holdout_pr_auc",
    ]
    for field in metadata_fields:
        assert f'"{field}"' in full_text
