"""Pydantic response schemas used by the recommendation API."""

from typing import Any, Literal

from pydantic import BaseModel, RootModel


class HealthResponse(BaseModel):
    status: str
    recommendations_loaded: bool


class RecommendedArticle(BaseModel):
    article_id: str
    rank: int


class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: list[RecommendedArticle]
    source: Literal["personalized", "popularity_fallback"]


class ModelInfoResponse(RootModel[dict[str, Any]]):
    """The persisted metadata object is returned without invented wrapper fields."""
