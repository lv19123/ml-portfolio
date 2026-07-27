from urllib.parse import parse_qs, urlparse


def is_youtube_url(url: str) -> bool:
    try:
        extract_youtube_video_id(url)
    except ValueError:
        return False
    return True


def extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()

    if hostname in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/shorts/", 1)[1].split("/", 1)[0]
        else:
            video_id = ""
    elif hostname == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
    else:
        video_id = ""

    if not video_id:
        raise ValueError("Invalid YouTube URL.")
    return video_id
