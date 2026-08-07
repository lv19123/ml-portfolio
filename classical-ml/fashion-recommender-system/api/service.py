"""In-memory lookup service for batch-generated recommendations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fashion_recommender.persistence import load_json, load_recommendations


class RecommendationService:
    """Load small serving artifacts once and answer requests by dictionary lookup."""

    def __init__(
        self,
        recommendations_path: str | Path,
        popular_items_path: str | Path,
        metadata_path: str | Path,
        *,
        strict: bool = True,
    ) -> None:
        self.recommendations_by_customer: dict[str, list[str]] = {}
        self.popular_items: list[str] = []
        self.model_metadata: dict[str, Any] = {}
        self.load_error: str | None = None
        try:
            recommendations = load_recommendations(recommendations_path)
            self.recommendations_by_customer = (
                recommendations.sort_values(["customer_id", "rank"])
                .groupby("customer_id", sort=False)["article_id"]
                .agg(lambda values: list(dict.fromkeys(values.astype(str))))
                .to_dict()
            )
            popular_value = load_json(popular_items_path)
            if isinstance(popular_value, dict):
                popular_value = popular_value.get("article_ids", popular_value.get("items", []))
            if not isinstance(popular_value, list):
                raise ValueError("popular_items.json must contain a list of article IDs")
            self.popular_items = list(dict.fromkeys(str(item) for item in popular_value))
            metadata_value = load_json(metadata_path)
            if not isinstance(metadata_value, dict):
                raise ValueError("model_metadata.json must contain a JSON object")
            self.model_metadata = metadata_value
        except (FileNotFoundError, ValueError, OSError) as error:
            self.load_error = str(error)
            if strict:
                raise

    @classmethod
    def from_project_root(
        cls,
        project_root: str | Path,
        *,
        strict: bool = True,
    ) -> "RecommendationService":
        root = Path(project_root).expanduser().resolve()
        return cls(
            root / "artifacts" / "final_recommendations.parquet",
            root / "models" / "popular_items.json",
            root / "models" / "model_metadata.json",
            strict=strict,
        )

    @property
    def recommendations_loaded(self) -> bool:
        return bool(self.recommendations_by_customer)

    def recommend(self, customer_id: str, k: int) -> tuple[list[dict[str, object]], str]:
        customer_id = str(customer_id)
        personalized = self.recommendations_by_customer.get(customer_id, [])
        source = "personalized" if personalized else "popularity_fallback"
        items = personalized if personalized else self.popular_items
        unique_items = list(dict.fromkeys(items))[:k]
        return (
            [
                {"article_id": article_id, "rank": rank}
                for rank, article_id in enumerate(unique_items, start=1)
            ],
            source,
        )
