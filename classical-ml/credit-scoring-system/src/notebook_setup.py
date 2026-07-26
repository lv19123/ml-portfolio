"""Общая настройка зависимостей и директорий для Jupyter-ноутбуков."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from src.config import (
    PROJECT_ROOT,
    ensure_project_directories,
    print_environment_info,
)


def install_if_missing(
    package_name: str,
    import_name: str | None = None,
) -> None:
    """Установить пакет, только если соответствующий модуль отсутствует."""
    module_name = import_name or package_name
    if importlib.util.find_spec(module_name) is not None:
        return

    print(f"Installing missing dependency: {package_name}")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            package_name,
        ]
    )


def setup_notebook(
    install_dependencies: bool = True,
) -> Path:
    """Подготовить окружение и вернуть абсолютный корень проекта."""
    if install_dependencies:
        dependencies = [
            ("catboost", None),
            ("joblib", None),
            ("matplotlib", None),
            ("ipython", "IPython"),
        ]
        for package_name, import_name in dependencies:
            install_if_missing(
                package_name,
                import_name,
            )

    ensure_project_directories()
    print_environment_info()
    return PROJECT_ROOT
