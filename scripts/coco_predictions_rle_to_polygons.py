#!/usr/bin/env python3
"""Convert COCO prediction JSON from RLE segmentation to polygon segmentation.

Requires ``pycocotools`` (install: ``pip install "sam3-prelabel-pipeline[coco]"`` or ``pip install pycocotools``).

SAM-3's ``PredictionDumper`` with ``iou_type: segm`` writes each prediction as::

  {"image_id", "category_id", "bbox", "score", "segmentation": <RLE dict>, "area"}

COCO also allows ``segmentation`` to be a list of polygons (each a flat [x,y,...]).
This script decodes RLE masks, extracts outer contours, and writes a new JSON file
with ``segmentation`` as a list of polygons while preserving ``bbox`` and ``score``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import pycocotools.mask as mask_utils
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This script requires pycocotools. Install with: pip install pycocotools"
    ) from e


def _rle_to_mask(seg: dict[str, Any] | list[Any]) -> np.ndarray:
    if isinstance(seg, dict):
        return mask_utils.decode(seg).astype(np.uint8)
    if isinstance(seg, list) and seg and isinstance(seg[0], list):
        # Already polygon format — rasterize for contour consistency
        raise ValueError("segmentation is already polygon list; nothing to convert")
    raise TypeError(f"Unsupported segmentation type: {type(seg)}")


def _mask_to_polygons(mask: np.ndarray, epsilon: float) -> list[list[float]]:
    mask_u8 = (mask.astype(np.uint8) * 255).copy()
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        approx = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
        if approx.shape[0] < 3:
            continue
        polygon = approx.reshape(-1, 2).astype(float).flatten().tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def convert_record(rec: dict[str, Any], epsilon: float) -> dict[str, Any]:
    out = dict(rec)
    seg = rec.get("segmentation")
    if seg is None:
        return out
    if isinstance(seg, list) and (not seg or isinstance(seg[0], list)):
        return out
    mask = _rle_to_mask(seg)  # type: ignore[arg-type]
    polys = _mask_to_polygons(mask, epsilon=epsilon)
    if polys:
        out["segmentation"] = polys
    else:
        out["segmentation"] = []
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="coco_predictions_segm.json")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=2.0,
        help="cv2.approxPolyDP epsilon in pixels (same role as pipeline polygon_approx_epsilon)",
    )
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    if not isinstance(data, list):
        raise SystemExit("Expected top-level JSON array of detections")

    converted = [convert_record(r, args.epsilon) for r in data]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, indent=2), encoding="utf-8")
    print(f"Wrote {len(converted)} records to {args.output}")


if __name__ == "__main__":
    main()
