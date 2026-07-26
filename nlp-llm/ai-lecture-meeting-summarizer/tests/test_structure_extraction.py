import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_openrouter_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)


def upload_txt(content: str = "Transformers use attention for sequence modeling.") -> dict:
    response = client.post(
        "/materials/upload",
        files={"file": ("lecture.txt", content.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    return response.json()


def upload_and_process_txt(
    content: str = "Transformers use attention for sequence modeling.",
) -> dict:
    metadata = upload_txt(content)
    response = client.post(f"/materials/{metadata['material_id']}/process")
    assert response.status_code == 200
    return response.json()


def test_generate_topics_for_processed_txt() -> None:
    metadata = upload_and_process_txt()

    response = client.post(f"/materials/{metadata['material_id']}/topics/generate")

    assert response.status_code == 200
    data = response.json()
    assert data["topics_count"] > 0
    assert isinstance(data["topics"], list)
    first_topic = data["topics"][0]
    assert first_topic["title"]
    assert first_topic["summary"]
    assert first_topic["source_start"]
    assert first_topic["source_end"]


def test_get_generated_topics() -> None:
    metadata = upload_and_process_txt()
    client.post(f"/materials/{metadata['material_id']}/topics/generate")

    response = client.get(f"/materials/{metadata['material_id']}/topics")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_generate_terms_for_processed_txt() -> None:
    metadata = upload_and_process_txt(
        "Transformers use attention mechanisms and embeddings for language modeling."
    )

    response = client.post(f"/materials/{metadata['material_id']}/terms/generate")

    assert response.status_code == 200
    data = response.json()
    assert data["terms_count"] > 0
    assert isinstance(data["terms"], list)
    first_term = data["terms"][0]
    assert first_term["term"]
    assert first_term["definition"]
    assert first_term["source"]


def test_get_generated_terms() -> None:
    metadata = upload_and_process_txt(
        "Transformers use attention mechanisms and embeddings for language modeling."
    )
    client.post(f"/materials/{metadata['material_id']}/terms/generate")

    response = client.get(f"/materials/{metadata['material_id']}/terms")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_topic_generation_before_processing_returns_400() -> None:
    metadata = upload_txt()

    response = client.post(f"/materials/{metadata['material_id']}/topics/generate")

    assert response.status_code == 400


def test_term_generation_before_processing_returns_400() -> None:
    metadata = upload_txt()

    response = client.post(f"/materials/{metadata['material_id']}/terms/generate")

    assert response.status_code == 400
