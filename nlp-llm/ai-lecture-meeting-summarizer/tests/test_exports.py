from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_openrouter_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)


def upload_txt(content: str = "Lecture text about exports.") -> dict:
    response = client.post(
        "/materials/upload",
        files={"file": ("lecture.txt", content.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    return response.json()


def upload_and_process_txt(content: str = "Lecture text about exports.") -> dict:
    metadata = upload_txt(content)
    response = client.post(f"/materials/{metadata['material_id']}/process")
    assert response.status_code == 200
    return response.json()


def test_create_pdf_export_for_short_report() -> None:
    metadata = upload_and_process_txt()
    client.post(f"/materials/{metadata['material_id']}/reports/short")

    response = client.post(
        f"/materials/{metadata['material_id']}/exports/pdf",
        params={"report_type": "short"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["format"] == "pdf"
    assert Path(data["file_path"]).exists()


def test_download_pdf_export_for_short_report() -> None:
    metadata = upload_and_process_txt()
    client.post(f"/materials/{metadata['material_id']}/reports/short")

    response = client.get(
        f"/materials/{metadata['material_id']}/download/pdf",
        params={"report_type": "short"},
    )

    assert response.status_code == 200
    assert "application/pdf" in response.headers["content-type"]


def test_create_docx_export_for_full_clean_report() -> None:
    metadata = upload_and_process_txt()
    client.post(f"/materials/{metadata['material_id']}/reports/full-clean")

    response = client.post(
        f"/materials/{metadata['material_id']}/exports/docx",
        params={"report_type": "full-clean"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["format"] == "docx"
    assert Path(data["file_path"]).exists()


def test_download_docx_export_for_full_clean_report() -> None:
    metadata = upload_and_process_txt()
    client.post(f"/materials/{metadata['material_id']}/reports/full-clean")

    response = client.get(
        f"/materials/{metadata['material_id']}/download/docx",
        params={"report_type": "full-clean"},
    )

    assert response.status_code == 200
    content_type = response.headers["content-type"]
    assert (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        in content_type
        or "application/octet-stream" in content_type
    )


def test_pdf_export_before_report_generation_returns_clear_error() -> None:
    metadata = upload_and_process_txt()

    response = client.post(
        f"/materials/{metadata['material_id']}/exports/pdf",
        params={"report_type": "short"},
    )

    assert response.status_code in {400, 404}
    assert "generate the report first" in response.json()["detail"].lower()
