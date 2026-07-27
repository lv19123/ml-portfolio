from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def upload_file(filename: str, content: bytes, content_type: str) -> dict:
    response = client.post(
        "/materials/upload",
        files={"file": (filename, content, content_type)},
    )
    assert response.status_code == 200
    return response.json()


def test_process_txt_file_returns_processed_metadata() -> None:
    metadata = upload_file(
        "lecture.txt",
        b"Lecture notes about transformers and attention.",
        "text/plain",
    )

    response = client.post(f"/materials/{metadata['material_id']}/process")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processed"
    assert data["segments_count"] > 0
    assert data["characters_count"] > 0


def test_processed_txt_file_text_contains_original_text() -> None:
    original_text = "Meeting notes about action items and deadlines."
    metadata = upload_file("meeting.txt", original_text.encode("utf-8"), "text/plain")
    client.post(f"/materials/{metadata['material_id']}/process")

    response = client.get(f"/materials/{metadata['material_id']}/text")

    assert response.status_code == 200
    assert original_text in response.json()["text"]


def test_processed_txt_file_segments_are_returned() -> None:
    metadata = upload_file(
        "seminar.txt",
        b"Seminar notes about embeddings.",
        "text/plain",
    )
    client.post(f"/materials/{metadata['material_id']}/process")

    response = client.get(f"/materials/{metadata['material_id']}/segments")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert data[0]["text"]
    assert data[0]["source"]

