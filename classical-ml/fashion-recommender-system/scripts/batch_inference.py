"""Command-line entry point for periodic recommendation recalculation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from fashion_recommender.batch import run_batch_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--recommendation-size", type=int, default=12)
    parser.add_argument("--user-batch-size", type=int, default=2000)
    arguments = parser.parse_args()
    run_batch_inference(
        arguments.project_root,
        recommendation_size=arguments.recommendation_size,
        user_batch_size=arguments.user_batch_size,
        collect_results=False,
    )
    output_path = arguments.project_root / "artifacts" / "final_recommendations.parquet"
    row_count = pq.ParquetFile(output_path).metadata.num_rows
    print(
        f"Saved {row_count:,} recommendation rows to {output_path}."
    )


if __name__ == "__main__":
    main()
