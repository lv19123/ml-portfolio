"""Run a local smoke demo against an already running backend.

Usage:
    python scripts/smoke_demo.py
"""

import os
import sys
from pathlib import Path

import requests


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
SAMPLE_PATH = Path("sample_data/ml_lecture_sample.txt")


def request_or_exit(method: str, path: str, **kwargs) -> requests.Response:
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        print(f"Request failed: {method} {path}")
        print(f"Backend URL: {BACKEND_URL}")
        print(f"Error: {exc}")
        sys.exit(1)


def main() -> None:
    if not SAMPLE_PATH.exists():
        print(f"Sample file not found: {SAMPLE_PATH}")
        sys.exit(1)

    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5)
        health.raise_for_status()
    except requests.RequestException:
        print(f"Backend is not reachable at {BACKEND_URL}. Start it first.")
        print("Example: uvicorn backend.app.main:app --reload")
        sys.exit(1)

    print("Uploading sample lecture...")
    with SAMPLE_PATH.open("rb") as file:
        upload = request_or_exit(
            "POST",
            "/materials/upload",
            files={"file": (SAMPLE_PATH.name, file, "text/plain")},
        ).json()
    material_id = upload["material_id"]
    print(f"material_id: {material_id}")

    print("Processing material...")
    request_or_exit("POST", f"/materials/{material_id}/process")

    print("Generating short report...")
    request_or_exit("POST", f"/materials/{material_id}/reports/short")

    print("Generating full cleaned notes...")
    request_or_exit("POST", f"/materials/{material_id}/reports/full-clean")

    print("Generating topics...")
    request_or_exit("POST", f"/materials/{material_id}/topics/generate")

    print("Generating key terms...")
    request_or_exit("POST", f"/materials/{material_id}/terms/generate")

    print("Building RAG index...")
    request_or_exit("POST", f"/materials/{material_id}/rag/build")

    print("Asking a question...")
    answer = request_or_exit(
        "POST",
        f"/materials/{material_id}/ask",
        json={"question": "Для чего используется линейная регрессия?", "top_k": 4},
    ).json()
    print(answer["answer"])

    print("Downloading Markdown short report...")
    markdown = request_or_exit(
        "GET",
        f"/materials/{material_id}/download/md",
        params={"report_type": "short"},
    )
    output_path = Path("data/reports") / material_id / "smoke_short_report_download.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown.text, encoding="utf-8")
    print(f"Saved: {output_path}")

    print(f"Smoke demo complete. material_id: {material_id}")


if __name__ == "__main__":
    main()
