import pytest

from src import model_config


def configure_environment(
    monkeypatch,
    *,
    in_colab,
    gpu_count,
):
    monkeypatch.setattr(
        model_config,
        "IN_COLAB",
        in_colab,
    )
    monkeypatch.setattr(
        model_config,
        "get_gpu_device_count",
        lambda: gpu_count,
    )


def test_colab_with_gpu_uses_first_gpu(monkeypatch):
    configure_environment(
        monkeypatch,
        in_colab=True,
        gpu_count=1,
    )

    assert model_config.get_catboost_device_config() == {
        "task_type": "GPU",
        "devices": "0",
    }


def test_colab_without_gpu_stops_with_instruction(monkeypatch):
    configure_environment(
        monkeypatch,
        in_colab=True,
        gpu_count=0,
    )

    with pytest.raises(
        RuntimeError,
        match="T4 GPU",
    ):
        model_config.get_catboost_device_config()


def test_local_with_gpu_uses_first_gpu(monkeypatch):
    configure_environment(
        monkeypatch,
        in_colab=False,
        gpu_count=2,
    )

    assert model_config.get_catboost_device_config() == {
        "task_type": "GPU",
        "devices": "0",
    }


def test_local_without_gpu_uses_cpu_fallback(monkeypatch):
    configure_environment(
        monkeypatch,
        in_colab=False,
        gpu_count=0,
    )

    assert model_config.get_catboost_device_config() == {
        "task_type": "CPU",
    }
