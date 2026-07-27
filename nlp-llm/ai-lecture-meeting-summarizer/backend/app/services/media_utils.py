import subprocess
from pathlib import Path

from backend.app.services.file_utils import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS


def is_audio_extension(extension: str) -> bool:
    return extension.lower() in AUDIO_EXTENSIONS


def is_video_extension(extension: str) -> bool:
    return extension.lower() in VIDEO_EXTENSIONS


def ensure_audio_path(input_path: Path, extension: str, output_dir: Path) -> Path:
    normalized_extension = extension.lower()
    if is_audio_extension(normalized_extension):
        return input_path

    if not is_video_extension(normalized_extension):
        raise ValueError(f"Unsupported media extension: {extension}")

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "audio.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "Could not extract audio from video. Make sure ffmpeg is installed."
        ) from exc

    return audio_path
