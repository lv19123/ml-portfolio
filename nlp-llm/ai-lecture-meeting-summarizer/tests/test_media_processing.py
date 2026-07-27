import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app
from backend.app.services.time_utils import format_seconds


client = TestClient(app)


@pytest.fixture(autouse=True)
def enable_fake_transcriber(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "STT_USE_FAKE_TRANSCRIBER", True)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)


def upload_fake_mp3() -> dict:
    response = client.post(
        "/materials/upload",
        files={"file": ("lecture.mp3", b"fake mp3 content", "audio/mpeg")},
    )
    assert response.status_code == 200
    return response.json()


def upload_and_process_fake_mp3() -> dict:
    metadata = upload_fake_mp3()
    response = client.post(f"/materials/{metadata['material_id']}/process")
    assert response.status_code == 200
    return response.json()


def test_format_seconds() -> None:
    assert format_seconds(0) == "00:00:00"
    assert format_seconds(65) == "00:01:05"
    assert format_seconds(3661) == "01:01:01"


def test_process_fake_mp3_with_timestamps() -> None:
    data = upload_and_process_fake_mp3()

    assert data["status"] == "processed"
    assert data["source_type"] == "audio"
    assert data["has_timestamps"] is True
    assert data["transcription_engine"] == "fake"
    assert data["segments_count"] > 0


def test_processed_fake_mp3_segments_include_timestamps() -> None:
    metadata = upload_and_process_fake_mp3()

    response = client.get(f"/materials/{metadata['material_id']}/segments")

    assert response.status_code == 200
    first_segment = response.json()[0]
    assert first_segment["text"]
    assert first_segment["start"]
    assert first_segment["end"]
    assert first_segment["source"]


def test_processed_fake_mp3_text_contains_fake_transcription() -> None:
    metadata = upload_and_process_fake_mp3()

    response = client.get(f"/materials/{metadata['material_id']}/text")

    assert response.status_code == 200
    assert "Это тестовая расшифровка аудио материала." in response.json()["text"]


def test_fake_mp3_short_report_generation_still_works() -> None:
    metadata = upload_and_process_fake_mp3()

    response = client.post(f"/materials/{metadata['material_id']}/reports/short")

    assert response.status_code == 200
    assert response.json()["report_type"] == "short"


def test_fake_mp3_rag_still_works() -> None:
    metadata = upload_and_process_fake_mp3()
    build_response = client.post(f"/materials/{metadata['material_id']}/rag/build")
    assert build_response.status_code == 200

    response = client.post(
        f"/materials/{metadata['material_id']}/ask",
        json={"question": "Что сказано в аудио материале?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"]
    assert isinstance(data["sources"], list)
