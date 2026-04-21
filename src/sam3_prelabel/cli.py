from __future__ import annotations

import argparse
import sys

from .config import load_config
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM-3 image pre-labeling pipeline and export COCO annotations."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (see config/pipeline.example.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    stats = run_pipeline(config)

    print("Pre-labeling completed.")
    print(f"Images processed: {stats.total_images}")
    print(f"Annotations exported: {stats.total_annotations}")
    print(f"Skipped (low score): {stats.skipped_low_score}")
    print(f"Skipped (small area): {stats.skipped_small_area}")
    print(f"Skipped (empty mask): {stats.skipped_empty_masks}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        raise
