from backend.app.config import settings
from backend.app.services.time_utils import format_seconds
from backend.app.services.youtube_utils import extract_youtube_video_id


def get_youtube_transcription_engine() -> str:
    if settings.YOUTUBE_USE_FAKE_TRANSCRIPT:
        return "fake-youtube-transcript"
    return "youtube-transcript-api"


def _fake_transcript() -> list[dict]:
    return [
        {
            "text": "Это тестовая расшифровка YouTube лекции.",
            "start": "00:00:00",
            "end": "00:00:10",
            "source": "00:00:00-00:00:10",
        },
        {
            "text": "В этом фрагменте объясняется линейная регрессия.",
            "start": "00:00:10",
            "end": "00:00:20",
            "source": "00:00:10-00:00:20",
        },
    ]


def _configured_languages() -> list[str]:
    return [
        language.strip()
        for language in settings.YOUTUBE_LANGUAGES.split(",")
        if language.strip()
    ] or ["ru", "en"]


def _item_value(item: object, key: str, default: object = "") -> object:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def fetch_youtube_transcript(url: str) -> list[dict]:
    if settings.YOUTUBE_USE_FAKE_TRANSCRIPT:
        return _fake_transcript()

    video_id = extract_youtube_video_id(url)

    try:
        from youtube_transcript_api import (
            NoTranscriptFound,
            TranscriptsDisabled,
            YouTubeTranscriptApi,
        )

        transcript_items = YouTubeTranscriptApi().fetch(
            video_id,
            languages=_configured_languages(),
        )
    except (NoTranscriptFound, TranscriptsDisabled) as exc:
        raise ValueError("No available transcript found for this YouTube video.") from exc
    except Exception as exc:
        raise ValueError("No available transcript found for this YouTube video.") from exc

    segments = []
    for item in transcript_items:
        text = str(_item_value(item, "text", "")).strip()
        if not text:
            continue

        start_seconds = float(_item_value(item, "start", 0))
        duration = float(_item_value(item, "duration", 0))
        end_seconds = start_seconds + duration
        start = format_seconds(start_seconds)
        end = format_seconds(end_seconds)
        segments.append(
            {
                "text": text,
                "start": start,
                "end": end,
                "source": f"{start}-{end}",
            }
        )

    if not segments:
        raise ValueError("No available transcript found for this YouTube video.")
    return segments
