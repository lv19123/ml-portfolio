"""Общая конфигурация вычислительного устройства CatBoost."""

from __future__ import annotations

from typing import Any

from catboost.utils import get_gpu_device_count

from src.config import IN_COLAB


COLAB_GPU_ERROR = (
    "GPU не обнаружена. В Google Colab выберите:\n"
    "Runtime → Change runtime type → T4 GPU,\n"
    "после чего перезапустите среду и выполните ноутбук заново."
)


def get_catboost_gpu_count() -> int:
    """Вернуть число GPU, доступных установленной сборке CatBoost."""
    try:
        return max(0, int(get_gpu_device_count()))
    except Exception:
        return 0


def get_catboost_device_config(
    *,
    in_colab: bool | None = None,
    gpu_count: int | None = None,
) -> dict[str, Any]:
    """Выбрать GPU или допустимый только локально CPU fallback."""
    is_colab = IN_COLAB if in_colab is None else in_colab
    available_gpu_count = (
        get_catboost_gpu_count()
        if gpu_count is None
        else max(0, int(gpu_count))
    )

    if available_gpu_count > 0:
        return {
            "task_type": "GPU",
            "devices": "0",
        }

    if is_colab:
        raise RuntimeError(COLAB_GPU_ERROR)

    return {
        "task_type": "CPU",
    }


def print_catboost_device_info(
    device_config: dict[str, Any],
    *,
    gpu_count: int,
    in_colab: bool | None = None,
) -> None:
    """Вывести краткую диагностику среды и выбранного устройства."""
    is_colab = IN_COLAB if in_colab is None else in_colab
    print(
        "Modeling environment:",
        "Google Colab" if is_colab else "local",
    )
    print(f"CatBoost GPU count: {int(gpu_count)}")
    print(
        "CatBoost device:",
        device_config["task_type"],
    )
    if device_config["task_type"] == "GPU":
        print(
            "CatBoost GPU devices:",
            device_config["devices"],
        )
