"""Единая конфигурация путей для локального запуска и Google Colab."""

from __future__ import annotations

import sys
from pathlib import Path


IN_COLAB = "google.colab" in sys.modules

# Измените только эту строку, если проект лежит в другой папке Google Drive.
COLAB_PROJECT_DIR = Path(
    "/content/drive/MyDrive/credit-scoring-system"
)

_CONTENT_PROJECT_DIR = Path("/content/credit-scoring-system")
_PROJECT_MARKERS = ("src", "notebooks", "data")


def _is_project_root(path: Path) -> bool:
    """Проверить наличие обязательных директорий репозитория."""
    return path.is_dir() and all(
        (path / marker).is_dir()
        for marker in _PROJECT_MARKERS
    )


def _path_and_parents(path: Path) -> list[Path]:
    """Вернуть путь и всех его родителей, начиная с ближайшего."""
    resolved_path = path.expanduser().resolve()
    if resolved_path.is_file():
        resolved_path = resolved_path.parent
    return [resolved_path, *resolved_path.parents]


def _mount_google_drive() -> None:
    """Подключить Google Drive только внутри Google Colab."""
    if not IN_COLAB:
        return

    drive_root = Path("/content/drive/MyDrive")
    if drive_root.is_dir():
        return

    try:
        from google.colab import drive
    except ImportError as error:
        raise RuntimeError(
            "Среда определена как Google Colab, но модуль "
            "google.colab недоступен."
        ) from error

    drive.mount("/content/drive")


def get_project_root() -> Path:
    """Найти корень проекта независимо от текущей рабочей директории."""
    checked_paths: list[Path] = []

    def check_candidates(candidates: list[Path]) -> Path | None:
        for candidate in candidates:
            resolved_candidate = candidate.expanduser().resolve()
            if resolved_candidate in checked_paths:
                continue
            checked_paths.append(resolved_candidate)
            if _is_project_root(resolved_candidate):
                return resolved_candidate
        return None

    if IN_COLAB:
        colab_root = check_candidates(
            [
                _CONTENT_PROJECT_DIR,
                COLAB_PROJECT_DIR,
            ]
        )
        if colab_root is not None:
            return colab_root

    module_root = Path(__file__).resolve().parents[1]
    local_candidates = [
        *_path_and_parents(module_root),
        *_path_and_parents(Path.cwd()),
    ]
    local_root = check_candidates(local_candidates)
    if local_root is not None:
        return local_root

    if IN_COLAB:
        _mount_google_drive()
        drive_root = check_candidates([COLAB_PROJECT_DIR])
        if drive_root is not None:
            return drive_root

    checked_text = "\n".join(
        f"- {path}"
        for path in checked_paths
    )
    raise FileNotFoundError(
        "Не удалось найти корень credit-scoring-system. "
        "Ожидаются директории src/, notebooks/ и data/. "
        f"Проверены пути:\n{checked_text}"
    )


PROJECT_ROOT = get_project_root()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
CLIENT_SPLIT_PATH = PROCESSED_DATA_DIR / "client_split.csv"
DEFAULT_MODEL_BUNDLE_PATH = MODELS_DIR / "credit_scoring_model.joblib"
DEFAULT_MODEL_METADATA_PATH = MODELS_DIR / "credit_scoring_model_metadata.json"
LEGACY_LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression_baseline.joblib"
FINAL_EVALUATION_PATH = TABLES_DIR / "final_evaluation.csv"


def ensure_project_directories() -> None:
    """Создать все директории для данных и результатов проекта."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        TABLES_DIR,
        FIGURES_DIR,
    ]
    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def find_data_file(filename: str) -> Path:
    """Найти файл данных в поддерживаемых локальных и Colab-папках."""
    filename_path = Path(filename)
    if filename_path.name != filename or filename_path.is_absolute():
        raise ValueError(
            "find_data_file принимает только имя файла без директорий: "
            f"{filename}"
        )

    checked_paths = [
        RAW_DATA_DIR / filename,
        RAW_DATA_DIR / "home-credit-default-risk" / filename,
        PROCESSED_DATA_DIR / filename,
        INTERIM_DATA_DIR / filename,
        DATA_DIR / filename,
    ]
    for path in checked_paths:
        if path.is_file():
            return path

    checked_text = "\n".join(
        f"- {path}"
        for path in checked_paths
    )
    raise FileNotFoundError(
        f"Файл данных {filename!r} не найден. Проверены пути:\n"
        f"{checked_text}"
    )


def print_environment_info() -> None:
    """Вывести краткую диагностику текущей среды и путей."""
    environment_name = "Google Colab" if IN_COLAB else "local"
    print(f"Environment: {environment_name}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data: {RAW_DATA_DIR}")
    print(f"Models: {MODELS_DIR}")
    print(f"Reports: {REPORTS_DIR}")


ensure_project_directories()
