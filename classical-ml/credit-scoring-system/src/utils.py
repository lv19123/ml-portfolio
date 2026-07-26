"""Небольшие общие функции для CLI и артефактов."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import catboost
import joblib
import numpy
import pandas
import sklearn


def runtime_versions() -> dict[str, str]:
    """Вернуть версии библиотек, влияющих на модельный артефакт."""
    return {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "catboost": catboost.__version__,
        "joblib": joblib.__version__,
    }


def write_json_atomically(data: dict[str, Any], path: Path) -> Path:
    """Атомарно записать строгий JSON без нестандартных NaN."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}-",
        suffix=".json.tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(
                data,
                output,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            output.write("\n")
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target
