from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_openrouter_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)


def upload_txt(content: str = "Lecture text about neural networks.") -> dict:
    response = client.post(
        "/materials/upload",
        files={"file": ("lecture.txt", content.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    return response.json()


def upload_and_process_txt(content: str = "Lecture text about neural networks.") -> dict:
    metadata = upload_txt(content)
    response = client.post(f"/materials/{metadata['material_id']}/process")
    assert response.status_code == 200
    return response.json()


def test_create_short_report_saves_markdown_file() -> None:
    metadata = upload_and_process_txt()

    response = client.post(f"/materials/{metadata['material_id']}/reports/short")

    assert response.status_code == 200
    data = response.json()
    assert data["report_type"] == "short"
    assert Path(data["report_path"]).exists()


def test_get_short_report_returns_markdown_content() -> None:
    metadata = upload_and_process_txt()
    client.post(f"/materials/{metadata['material_id']}/reports/short")

    response = client.get(f"/materials/{metadata['material_id']}/reports/short")

    assert response.status_code == 200
    assert "Краткий отчёт" in response.json()["content"]


def test_create_full_clean_report_saves_markdown_file() -> None:
    metadata = upload_and_process_txt()

    response = client.post(f"/materials/{metadata['material_id']}/reports/full-clean")

    assert response.status_code == 200
    data = response.json()
    assert data["report_type"] == "full_clean"
    assert Path(data["report_path"]).exists()


def test_download_short_markdown_report() -> None:
    metadata = upload_and_process_txt()
    client.post(f"/materials/{metadata['material_id']}/reports/short")

    response = client.get(
        f"/materials/{metadata['material_id']}/download/md",
        params={"report_type": "short"},
    )

    assert response.status_code == 200
    content_type = response.headers["content-type"]
    assert "text/markdown" in content_type or "application/octet-stream" in content_type


def test_report_generation_before_processing_returns_400() -> None:
    metadata = upload_txt()

    response = client.post(f"/materials/{metadata['material_id']}/reports/short")

    assert response.status_code == 400
