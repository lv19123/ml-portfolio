"""Project configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv


# Project root: app/config.py -> app/ -> project directory.
SCRIPT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(SCRIPT_DIR / ".env")

MATERIALS_DIR = SCRIPT_DIR / "materials"
CHROMA_DIR = SCRIPT_DIR / "chroma_db"
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50
RAG_TOP_K = 5

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROUP = os.getenv("MIET_GROUP", "ИКТ-42")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}" if BOT_TOKEN else ""


def require_bot_token() -> str:
    """Возвращает Telegram token или падает только при реальном запуске бота."""
    if not BOT_TOKEN:
        raise RuntimeError("Нужно задать TELEGRAM_BOT_TOKEN в .env")
    return BOT_TOKEN


def get_base_url() -> str:
    """Возвращает Telegram Bot API base URL после проверки token."""
    require_bot_token()
    return f"https://api.telegram.org/bot{BOT_TOKEN}"
