from pathlib import Path

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
AUDIO_EXTENSIONS = {".mp3", ".wav"}
VIDEO_EXTENSIONS = {".mp4", ".mov"}
SUPPORTED_EXTENSIONS = DOCUMENT_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def detect_source_type(extension: str) -> str:
    normalized_extension = extension.lower()
    if normalized_extension in DOCUMENT_EXTENSIONS:
        return "document"
    if normalized_extension in AUDIO_EXTENSIONS:
        return "audio"
    if normalized_extension in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"Unsupported file extension: {extension}")


def validate_extension(extension: str) -> bool:
    return extension.lower() in SUPPORTED_EXTENSIONS
