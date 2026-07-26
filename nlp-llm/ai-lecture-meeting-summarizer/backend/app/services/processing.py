import json
from pathlib import Path
from typing import Any

from backend.app.config import ensure_directories, settings
from backend.app.services.document_extractor import extract_document
from backend.app.services.media_utils import ensure_audio_path
from backend.app.services.speech_to_text import get_transcription_engine, transcribe_audio
from backend.app.services.storage import get_material, load_materials, save_materials
from backend.app.services.youtube_transcript import (
    fetch_youtube_transcript,
    get_youtube_transcription_engine,
)


class MaterialNotFoundError(ValueError):
    pass


class UnsupportedProcessingError(ValueError):
    pass


class ProcessingError(ValueError):
    pass


def _save_material_metadata(metadata: dict[str, Any]) -> None:
    materials = load_materials()
    materials[metadata["material_id"]] = metadata
    save_materials(materials)


def _mark_failed(metadata: dict[str, Any], message: str) -> None:
    metadata["status"] = "failed"
    metadata["error_message"] = message
    _save_material_metadata(metadata)


def _save_processed_artifacts(
    metadata: dict[str, Any],
    segments: list[dict],
    processed_dir: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extracted_text = "\n\n".join(segment["text"] for segment in segments)
    if not extracted_text.strip():
        raise ProcessingError("No extractable text found")

    processed_dir.mkdir(parents=True, exist_ok=True)
    segments_path = processed_dir / "segments.json"
    extracted_text_path = processed_dir / "extracted_text.txt"

    with segments_path.open("w", encoding="utf-8") as file:
        json.dump(segments, file, indent=2, ensure_ascii=False)
    extracted_text_path.write_text(extracted_text, encoding="utf-8")

    metadata.update(
        {
            "status": "processed",
            "processed_dir": str(processed_dir),
            "segments_path": str(segments_path),
            "extracted_text_path": str(extracted_text_path),
            "segments_count": len(segments),
            "characters_count": len(extracted_text),
            "error_message": None,
        }
    )
    if extra_metadata:
        metadata.update(extra_metadata)

    _save_material_metadata(metadata)
    return metadata


def process_material(material_id: str) -> dict[str, Any]:
    metadata = get_material(material_id)
    if metadata is None:
        raise MaterialNotFoundError("Material not found")

    try:
        ensure_directories()
        processed_dir = Path(settings.PROCESSED_DIR) / material_id

        if metadata["source_type"] == "document":
            upload_path = Path(settings.UPLOADS_DIR) / str(metadata["stored_filename"])
            if not upload_path.exists():
                raise ProcessingError("Uploaded file was not found")

            segments = extract_document(upload_path, metadata["file_extension"])
            return _save_processed_artifacts(
                metadata,
                segments,
                processed_dir,
                {"has_timestamps": False},
            )

        if metadata["source_type"] in {"audio", "video"}:
            upload_path = Path(settings.UPLOADS_DIR) / str(metadata["stored_filename"])
            if not upload_path.exists():
                raise ProcessingError("Uploaded file was not found")

            audio_path = ensure_audio_path(upload_path, metadata["file_extension"], processed_dir)
            segments = transcribe_audio(audio_path)
            return _save_processed_artifacts(
                metadata,
                segments,
                processed_dir,
                {
                    "has_timestamps": True,
                    "timestamp_format": "HH:MM:SS",
                    "transcription_engine": get_transcription_engine(),
                },
            )

        if metadata["source_type"] == "youtube":
            source_url = metadata.get("source_url")
            if not source_url:
                raise ProcessingError("YouTube material has no source URL")

            segments = fetch_youtube_transcript(source_url)
            return _save_processed_artifacts(
                metadata,
                segments,
                processed_dir,
                {
                    "has_timestamps": True,
                    "timestamp_format": "HH:MM:SS",
                    "transcription_engine": get_youtube_transcription_engine(),
                },
            )

        raise UnsupportedProcessingError(
            f"Unsupported source type: {metadata['source_type']}"
        )
    except (MaterialNotFoundError, UnsupportedProcessingError):
        raise
    except Exception as exc:
        message = str(exc) or "Material processing failed"
        _mark_failed(metadata, message)
        raise ProcessingError(message) from exc
