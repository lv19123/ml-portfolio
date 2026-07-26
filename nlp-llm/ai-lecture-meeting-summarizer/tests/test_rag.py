import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_openrouter_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)


def upload_txt(
    content: str = "Линейная регрессия используется для предсказания числовых значений.",
) -> dict:
    response = client.post(
        "/materials/upload",
        files={"file": ("lecture.txt", content.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    return response.json()


def upload_and_process_txt(
    content: str = "Линейная регрессия используется для предсказания числовых значений.",
) -> dict:
    metadata = upload_txt(content)
    response = client.post(f"/materials/{metadata['material_id']}/process")
    assert response.status_code == 200
    return response.json()


def test_build_rag_index_for_processed_txt() -> None:
    metadata = upload_and_process_txt()

    response = client.post(f"/materials/{metadata['material_id']}/rag/build")

    assert response.status_code == 200
    data = response.json()
    assert data["chunks_count"] > 0
    assert data["retriever"] == "tfidf"


def test_get_rag_status_after_build() -> None:
    metadata = upload_and_process_txt()
    client.post(f"/materials/{metadata['material_id']}/rag/build")

    response = client.get(f"/materials/{metadata['material_id']}/rag/status")

    assert response.status_code == 200
    data = response.json()
    assert data["rag_ready"] is True
    assert data["chunks_count"] > 0


def test_ask_question_with_existing_rag_index() -> None:
    metadata = upload_and_process_txt()
    client.post(f"/materials/{metadata['material_id']}/rag/build")

    response = client.post(
        f"/materials/{metadata['material_id']}/ask",
        json={"question": "Для чего используется линейная регрессия?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"]
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0


def test_ask_question_auto_builds_rag_index() -> None:
    metadata = upload_and_process_txt()

    response = client.post(
        f"/materials/{metadata['material_id']}/ask",
        json={"question": "Что предсказывает линейная регрессия?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"]


def test_rag_build_before_processing_returns_400() -> None:
    metadata = upload_txt()

    response = client.post(f"/materials/{metadata['material_id']}/rag/build")

    assert response.status_code == 400


def test_ask_empty_question_returns_400() -> None:
    metadata = upload_and_process_txt()

    response = client.post(
        f"/materials/{metadata['material_id']}/ask",
        json={"question": "   "},
    )

    assert response.status_code == 400
