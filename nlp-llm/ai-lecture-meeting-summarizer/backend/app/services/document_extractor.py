from pathlib import Path

from docx import Document
from pypdf import PdfReader


def _clean_segments(segments: list[dict]) -> list[dict]:
    cleaned_segments = []
    for segment in segments:
        text = segment["text"].strip()
        if text:
            cleaned_segments.append({**segment, "text": text})
    return cleaned_segments


def _require_segments(segments: list[dict], file_type: str) -> list[dict]:
    cleaned_segments = _clean_segments(segments)
    if not cleaned_segments:
        raise ValueError(f"No extractable text found in {file_type} file")
    return cleaned_segments


def extract_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    return _require_segments([{"text": text, "source": "text"}], "TXT")


def extract_markdown(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    return _require_segments([{"text": text, "source": "text"}], "Markdown")


def extract_pdf(path: Path) -> list[dict]:
    reader = PdfReader(path)
    segments = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        segments.append({"text": text, "source": f"page {page_number}"})
    return _require_segments(segments, "PDF")


def extract_docx(path: Path) -> list[dict]:
    document = Document(path)
    segments = []
    paragraph_number = 1
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            segments.append({"text": text, "source": f"paragraph {paragraph_number}"})
            paragraph_number += 1
    return _require_segments(segments, "DOCX")


def extract_document(path: Path, extension: str) -> list[dict]:
    normalized_extension = extension.lower()
    if normalized_extension == ".txt":
        return extract_txt(path)
    if normalized_extension == ".md":
        return extract_markdown(path)
    if normalized_extension == ".pdf":
        return extract_pdf(path)
    if normalized_extension == ".docx":
        return extract_docx(path)
    raise ValueError(f"Unsupported document extension for extraction: {extension}")
