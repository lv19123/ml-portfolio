import importlib
from pathlib import Path

import pytest

import src
from src import config
from src import experiment_tracking
from src import model_config
from src import notebook_setup


def test_project_root_does_not_depend_on_current_directory(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    assert config.get_project_root() == Path(__file__).resolve().parents[1]


def test_ensure_project_directories_creates_expected_tree(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    directories = {
        "DATA_DIR": data_dir,
        "RAW_DATA_DIR": data_dir / "raw",
        "INTERIM_DATA_DIR": data_dir / "interim",
        "PROCESSED_DATA_DIR": data_dir / "processed",
        "MODELS_DIR": tmp_path / "models",
        "REPORTS_DIR": tmp_path / "reports",
        "TABLES_DIR": tmp_path / "reports" / "tables",
        "FIGURES_DIR": tmp_path / "reports" / "figures",
    }
    for name, path in directories.items():
        monkeypatch.setattr(config, name, path)

    config.ensure_project_directories()

    for path in directories.values():
        assert path.is_dir()


def test_find_data_file_checks_supported_locations(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    raw_data_dir = data_dir / "raw"
    nested_raw_dir = raw_data_dir / "home-credit-default-risk"
    processed_data_dir = data_dir / "processed"
    interim_data_dir = data_dir / "interim"

    for directory in [
        raw_data_dir,
        nested_raw_dir,
        processed_data_dir,
        interim_data_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    expected_path = nested_raw_dir / "application_train.csv"
    expected_path.write_text("SK_ID_CURR,TARGET\n1,0\n", encoding="utf-8")

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "RAW_DATA_DIR", raw_data_dir)
    monkeypatch.setattr(
        config,
        "PROCESSED_DATA_DIR",
        processed_data_dir,
    )
    monkeypatch.setattr(
        config,
        "INTERIM_DATA_DIR",
        interim_data_dir,
    )

    assert config.find_data_file("application_train.csv") == expected_path


def test_find_data_file_error_lists_checked_paths(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "RAW_DATA_DIR", data_dir / "raw")
    monkeypatch.setattr(
        config,
        "PROCESSED_DATA_DIR",
        data_dir / "processed",
    )
    monkeypatch.setattr(
        config,
        "INTERIM_DATA_DIR",
        data_dir / "interim",
    )

    with pytest.raises(FileNotFoundError) as error:
        config.find_data_file("missing.csv")

    message = str(error.value)
    assert "missing.csv" in message
    assert "home-credit-default-risk" in message
    assert "processed" in message


def test_src_package_and_modules_import():
    assert src.__name__ == "src"
    assert importlib.import_module("src.config") is config
    assert importlib.import_module("src.features")
    assert importlib.import_module("src.validation")
    assert importlib.import_module("src.notebook_setup")
    assert (
        importlib.import_module("src.experiment_tracking")
        is experiment_tracking
    )
    assert importlib.import_module("src.model_config") is model_config


def test_install_if_missing_skips_installed_package(monkeypatch):
    calls = []
    monkeypatch.setattr(
        notebook_setup.importlib.util,
        "find_spec",
        lambda module_name: object(),
    )
    monkeypatch.setattr(
        notebook_setup.subprocess,
        "check_call",
        calls.append,
    )

    notebook_setup.install_if_missing("catboost")

    assert calls == []


def test_install_if_missing_uses_current_python(monkeypatch):
    calls = []
    monkeypatch.setattr(
        notebook_setup.importlib.util,
        "find_spec",
        lambda module_name: None,
    )
    monkeypatch.setattr(
        notebook_setup.subprocess,
        "check_call",
        calls.append,
    )

    notebook_setup.install_if_missing("catboost")

    assert calls == [
        [
            notebook_setup.sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "catboost",
        ]
    ]
