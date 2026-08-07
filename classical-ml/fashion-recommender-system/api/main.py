"""FastAPI entry point that serves only precomputed recommendations."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Query, Request

from api.schemas import HealthResponse, ModelInfoResponse, RecommendationResponse
from api.service import RecommendationService


def create_app(
    project_root: str | Path | None = None,
    service: RecommendationService | None = None,
) -> FastAPI:
    root = Path(
        project_root
        or os.environ.get("FASHION_PROJECT_ROOT")
        or Path(__file__).resolve().parents[1]
    )
    recommendation_service = service or RecommendationService.from_project_root(
        root,
        strict=False,
    )
    application = FastAPI(
        title="Fashion Recommender API",
        version="1.0.0",
        description="Lookup API for batch-generated Top-12 recommendations.",
    )
    application.state.recommendation_service = recommendation_service

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        current_service = request.app.state.recommendation_service
        loaded = current_service.recommendations_loaded
        return HealthResponse(
            status="ok" if loaded else "degraded",
            recommendations_loaded=loaded,
        )

    @application.get(
        "/recommend/{customer_id}",
        response_model=RecommendationResponse,
    )
    def recommend(
        customer_id: str,
        request: Request,
        k: int = Query(default=12, ge=1, le=50),
    ) -> RecommendationResponse:
        recommendations, source = request.app.state.recommendation_service.recommend(
            customer_id,
            k,
        )
        return RecommendationResponse(
            customer_id=customer_id,
            recommendations=recommendations,
            source=source,
        )

    @application.get("/model-info", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        return ModelInfoResponse(
            root=request.app.state.recommendation_service.model_metadata
        )

    return application


app = create_app()
