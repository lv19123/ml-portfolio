from pathlib import Path
from typing import Any

from backend.app.config import settings
from backend.app.services.time_utils import format_seconds

_whisper_model: Any | None = None


def get_transcription_engine() -> str:
    if settings.STT_USE_FAKE_TRANSCRIBER:
        return "fake"
    return "faster-whisper"


def _fake_transcription() -> list[dict]:
    return [
        {
            "text": "Это тестовая расшифровка аудио материала.",
            "start": "00:00:00",
            "end": "00:00:10",
            "source": "00:00:00-00:00:10",
        }
    ]


def _load_whisper_model() -> Any:
    global _whisper_model
    if _whisper_model is None:
        # Lazy import/load keeps app startup cheap and lets fake mode run without a model.
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(
            settings.STT_MODEL_SIZE,
            device=settings.STT_DEVICE,
            compute_type=settings.STT_COMPUTE_TYPE,
        )
    return _whisper_model


def transcribe_audio(audio_path: Path) -> list[dict]:
    if settings.STT_USE_FAKE_TRANSCRIBER:
        return _fake_transcription()

    model = _load_whisper_model()
    segments_iterator, _info = model.transcribe(str(audio_path))

    segments = []
    for segment in segments_iterator:
        text = segment.text.strip()
        if not text:
            continue

        start = format_seconds(segment.start)
        end = format_seconds(segment.end)
        segments.append(
            {
                "text": text,
                "start": start,
                "end": end,
                "source": f"{start}-{end}",
            }
        )

    if not segments:
        raise ValueError("Speech recognition produced no text.")
    return segments
