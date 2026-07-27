import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app
from backend.app.services.youtube_utils import extract_youtube_video_id


client = TestClient(app)
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture(autouse=True)
def enable_fake_youtube_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "YOUTUBE_USE_FAKE_TRANSCRIPT", True)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)


def add_youtube_material(url: str = YOUTUBE_URL) -> dict:
    response = client.post("/materials/youtube", json={"url": url})
    assert response.status_code == 200
    return response.json()


def add_and_process_youtube_material() -> dict:
    metadata = add_youtube_material()
    response = client.post(f"/materials/{metadata['material_id']}/process")
    assert response.status_code == 200
    return response.json()


def test_extract_youtube_video_id_from_watch_url() -> None:
    assert extract_youtube_video_id(YOUTUBE_URL) == "dQw4w9WgXcQ"


def test_extract_youtube_video_id_from_short_url() -> None:
    assert extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_youtube_video_id_from_shorts_url() -> None:
    assert (
        extract_youtube_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ")
        == "dQw4w9WgXcQ"
    )


def test_add_youtube_material() -> None:
    data = add_youtube_material()

    assert data["source_type"] == "youtube"
    assert data["status"] == "uploaded"
    assert data["source_url"] == YOUTUBE_URL
    assert data["material_id"]


def test_process_youtube_material_with_fake_transcript() -> None:
    data = add_and_process_youtube_material()

    assert data["status"] == "processed"
    assert data["has_timestamps"] is True
    assert data["transcription_engine"] == "fake-youtube-transcript"
    assert data["segments_count"] > 0


def test_youtube_segments_include_timestamps() -> None:
    metadata = add_and_process_youtube_material()

    response = client.get(f"/materials/{metadata['material_id']}/segments")

    assert response.status_code == 200
    first_segment = response.json()[0]
    assert first_segment["text"]
    assert first_segment["start"]
    assert first_segment["end"]
    assert first_segment["source"]


def test_youtube_text_contains_fake_transcript() -> None:
    metadata = add_and_process_youtube_material()

    response = client.get(f"/materials/{metadata['material_id']}/text")

    assert response.status_code == 200
    assert "Это тестовая расшифровка YouTube лекции." in response.json()["text"]


def test_youtube_short_report_generation_still_works() -> None:
    metadata = add_and_process_youtube_material()

    response = client.post(f"/materials/{metadata['material_id']}/reports/short")

    assert response.status_code == 200


def test_youtube_rag_still_works() -> None:
    metadata = add_and_process_youtube_material()
    build_response = client.post(f"/materials/{metadata['material_id']}/rag/build")
    assert build_response.status_code == 200

    response = client.post(
        f"/materials/{metadata['material_id']}/ask",
        json={"question": "Что объясняется в видео?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"]
    assert isinstance(data["sources"], list)


def test_add_youtube_material_with_invalid_url_returns_400() -> None:
    response = client.post("/materials/youtube", json={"url": "https://example.com/video"})

    assert response.status_code == 400
