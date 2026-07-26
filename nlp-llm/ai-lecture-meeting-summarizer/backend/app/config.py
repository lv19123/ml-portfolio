from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    DATA_DIR: Path = Path("data")
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    REPORTS_DIR: Path = DATA_DIR / "reports"
    VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"
    MATERIALS_FILE: Path = DATA_DIR / "materials.json"
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = "openai/gpt-4o-mini"
    STT_MODEL_SIZE: str = "tiny"
    STT_DEVICE: str = "cpu"
    STT_COMPUTE_TYPE: str = "int8"
    STT_USE_FAKE_TRANSCRIBER: bool = False
    YOUTUBE_USE_FAKE_TRANSCRIPT: bool = False
    YOUTUBE_LANGUAGES: str = "ru,en"


settings = Settings()


def ensure_directories() -> None:
    for directory in (
        settings.DATA_DIR,
        settings.UPLOADS_DIR,
        settings.PROCESSED_DIR,
        settings.REPORTS_DIR,
        settings.VECTOR_DB_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
