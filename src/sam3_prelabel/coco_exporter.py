from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    segmentation: list[list[float]]
    area: float
    bbox: list[float]
    iscrowd: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "segmentation": self.segmentation,
            "area": self.area,
            "bbox": self.bbox,
            "iscrowd": self.iscrowd,
        }


def binary_mask_to_polygons(
    mask: np.ndarray, epsilon: float = 2.0
) -> list[list[float]]:
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


def mask_to_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image_bgr.copy()
    alpha = 0.45
    out[mask > 0] = (
        (1 - alpha) * out[mask > 0] + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return out


def ensure_output_dirs(base_dir: Path, save_masks: bool, save_visualizations: bool) -> dict[str, Path]:
    dirs = {
        "base": base_dir,
        "annotations": base_dir / "annotations",
    }
    dirs["base"].mkdir(parents=True, exist_ok=True)
    dirs["annotations"].mkdir(parents=True, exist_ok=True)

    if save_masks:
        dirs["masks"] = base_dir / "masks"
        dirs["masks"].mkdir(parents=True, exist_ok=True)
    if save_visualizations:
        dirs["visualizations"] = base_dir / "visualizations"
        dirs["visualizations"].mkdir(parents=True, exist_ok=True)

    return dirs
