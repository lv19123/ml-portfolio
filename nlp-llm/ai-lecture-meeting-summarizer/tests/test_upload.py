from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_upload_txt_file_returns_metadata() -> None:
    response = client.post(
        "/materials/upload",
        files={"file": ("lecture.txt", b"These are lecture notes.", "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["material_id"]
    assert data["source_type"] == "document"
    assert data["status"] == "uploaded"
