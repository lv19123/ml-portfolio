from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.materials import router as materials_router
from backend.app.config import ensure_directories


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    yield


app = FastAPI(title="AI Lecture / Meeting Summarizer", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "ai-lecture-summarizer",
    }


app.include_router(materials_router)
