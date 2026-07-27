"""Быстрая проверка инфраструктурной совместимости с Google Colab."""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src
from src.config import (
    FIGURES_DIR,
    IN_COLAB,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT as CONFIG_PROJECT_ROOT,
    RAW_DATA_DIR,
    TABLES_DIR,
    ensure_project_directories,
    find_data_file,
)
from src.experiment_tracking import (
    MODEL_METRICS_COLUMNS,
    MODEL_METRICS_PATH,
)
from src.model_config import (
    COLAB_GPU_ERROR,
    get_catboost_device_config,
    get_catboost_gpu_count,
)


REQUIRED_DATA_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "HomeCredit_columns_description.csv",
    "sample_submission.csv",
]

WINDOWS_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z])[A-Za-z]:[\\/]"
)
FORBIDDEN_PATTERNS = {
    "жёсткий пользовательский путь": re.compile(
        "/" + r"Users/[^/\s]+/"
    ),
    "старый путь через cwd к data": re.compile(
        r"Path\.cwd\(\)\s*/\s*[\"']data[\"']"
    ),
    "относительный sys.path.append": re.compile(
        r"sys\.path\.append\(\s*[\"']\.\./"
    ),
}
MODELING_NOTEBOOKS = [
    "02_application_baseline.ipynb",
    "03_bureau_features.ipynb",
    "04_previous_application_features.ipynb",
    "05_installments_features.ipynb",
    "06_pos_cash_features.ipynb",
    "07_credit_card_features.ipynb",
    "08_final_model.ipynb",
]


def iter_project_sources() -> list[Path]:
    """Вернуть Python-файлы и ноутбуки без виртуальных окружений."""
    paths = [
        *PROJECT_ROOT.rglob("*.py"),
        *PROJECT_ROOT.rglob("*.ipynb"),
    ]
    excluded_parts = {
        ".git",
        ".venv",
        "__pycache__",
        ".ipynb_checkpoints",
    }
    return sorted(
        path
        for path in paths
        if excluded_parts.isdisjoint(path.parts)
    )


def read_source(path: Path) -> str:
    """Прочитать Python-код или объединить кодовые ячейки ноутбука."""
    if path.suffix == ".py":
        return path.read_text(encoding="utf-8")

    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def main(
    require_data: bool = True,
) -> int:
    """Выполнить проверки без обучения моделей."""
    problems: list[str] = []

    if CONFIG_PROJECT_ROOT != PROJECT_ROOT:
        problems.append(
            "src.config определил другой PROJECT_ROOT: "
            f"{CONFIG_PROJECT_ROOT}"
        )

    try:
        ensure_project_directories()
    except OSError as error:
        problems.append(
            f"Не удалось создать директории проекта: {error}"
        )

    required_directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        TABLES_DIR,
        FIGURES_DIR,
    ]
    for directory in required_directories:
        if not directory.is_dir():
            problems.append(
                f"Не существует директория: {directory}"
            )

    for writable_directory in [
        TABLES_DIR,
        MODELS_DIR,
    ]:
        write_test_path = (
            writable_directory
            / ".colab_write_test"
        )
        try:
            write_test_path.write_text(
                "ok",
                encoding="utf-8",
            )
            write_test_path.unlink()
        except OSError as error:
            problems.append(
                "Нет доступа на запись в "
                f"{writable_directory}: {error}"
            )

    for module_info in pkgutil.walk_packages(
        src.__path__,
        prefix="src.",
    ):
        try:
            importlib.import_module(module_info.name)
        except Exception as error:
            problems.append(
                f"Не импортируется {module_info.name}: "
                f"{type(error).__name__}: {error}"
            )

    try:
        import catboost
    except ImportError as error:
        problems.append(
            f"Не импортируется CatBoost: {error}"
        )
    else:
        print(f"CatBoost version: {catboost.__version__}")

    gpu_count = get_catboost_gpu_count()
    print(f"CatBoost GPU count: {gpu_count}")
    if IN_COLAB:
        try:
            device_config = get_catboost_device_config(
                in_colab=True,
                gpu_count=gpu_count,
            )
        except RuntimeError:
            problems.append(COLAB_GPU_ERROR)
        else:
            print(
                "CatBoost device:",
                device_config["task_type"],
            )
    else:
        device_config = get_catboost_device_config(
            in_colab=False,
            gpu_count=gpu_count,
        )
        print(
            "CatBoost device for local notebooks:",
            device_config["task_type"],
        )

    if MODEL_METRICS_PATH != TABLES_DIR / "model_metrics.csv":
        problems.append(
            "Единый путь метрик настроен неверно: "
            f"{MODEL_METRICS_PATH}"
        )
    if not MODEL_METRICS_COLUMNS:
        problems.append(
            "Схема model_metrics.csv пуста."
        )

    if require_data:
        for filename in REQUIRED_DATA_FILES:
            try:
                find_data_file(filename)
            except FileNotFoundError as error:
                problems.append(str(error))

    project_sources: dict[Path, str] = {}
    for path in iter_project_sources():
        source = read_source(path)
        project_sources[path] = source
        relative_path = path.relative_to(PROJECT_ROOT)

        if WINDOWS_PATH_PATTERN.search(source):
            problems.append(
                f"Жёсткий Windows-путь: {relative_path}"
            )

        for description, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(source):
                problems.append(
                    f"{description}: {relative_path}"
                )

    all_source_text = "\n".join(project_sources.values())
    legacy_report_names = [
        "baseline_model_" + "comparison.csv",
        "model_" + "comparison.csv",
    ]
    for legacy_name in legacy_report_names:
        if legacy_name in all_source_text:
            problems.append(
                f"Найдена ссылка на старый отчёт: {legacy_name}"
            )

    for notebook_name in MODELING_NOTEBOOKS:
        notebook_path = (
            PROJECT_ROOT
            / "notebooks"
            / notebook_name
        )
        notebook_source = project_sources.get(
            notebook_path,
            "",
        )
        if "save_experiment_result(" not in notebook_source:
            problems.append(
                f"{notebook_name} не записывает model_metrics.csv "
                "через общую функцию."
            )
        if notebook_name == "08_final_model.ipynb":
            if "get_catboost_gpu_count()" not in notebook_source:
                problems.append(
                    "08_final_model.ipynb не проверяет доступность GPU."
                )
            if "get_catboost_device_config(" in notebook_source:
                problems.append(
                    "08_final_model.ipynb использует конфигурацию, "
                    "запрещающую CPU fallback в Colab."
                )
            required_run_mode_fragments = [
                'REQUESTED_RUN_MODE = "auto"',
                '"full_gpu"',
                '"full_cpu"',
                '"cpu_debug"',
                "train_size=50_000",
                "train_size=15_000",
                '"iterations": MAX_ITERATIONS',
                (
                    'catboost_params["thread_count"] = '
                    "CPU_THREAD_COUNT"
                ),
                (
                    "early_stopping_rounds="
                    "EARLY_STOPPING_ROUNDS"
                ),
            ]
            for fragment in required_run_mode_fragments:
                if fragment not in notebook_source:
                    problems.append(
                        "08_final_model.ipynb не содержит "
                        f"обязательную настройку режима: {fragment}"
                    )
        elif "get_catboost_device_config(" not in notebook_source:
            problems.append(
                f"{notebook_name} не использует общую "
                "GPU-конфигурацию."
            )
        if notebook_source.count(
            "**catboost_device_config"
        ) != 2:
            problems.append(
                f"{notebook_name} не передаёт device config "
                "одновременно в catboost.cv и CatBoostClassifier."
            )

    final_notebook_path = (
        PROJECT_ROOT
        / "notebooks"
        / "08_final_model.ipynb"
    )
    final_source = project_sources.get(
        final_notebook_path,
        "",
    )
    required_final_fragments = [
        'MODELS_DIR / "catboost_all_features.cbm"',
        'MODELS_DIR / "catboost_all_features_metadata.json"',
        'MODELS_DIR / "catboost_all_features_cpu_debug.cbm"',
        '"catboost_all_features_cpu_debug_metadata.json"',
        "catboost_model.save_model(",
        "json.dump(",
    ]
    for fragment in required_final_fragments:
        if fragment not in final_source:
            problems.append(
                "08_final_model.ipynb не сохраняет обязательный "
                f"артефакт: {fragment}"
            )

    if problems:
        print("Google Colab compatibility problems found:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Google Colab compatibility check passed")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help=(
            "Проверить инфраструктуру без обязательного наличия "
            "приватных CSV."
        ),
    )
    arguments = parser.parse_args()
    raise SystemExit(
        main(require_data=not arguments.skip_data)
    )
