from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class RoboflowPrediction:
    masks: np.ndarray
    scores: np.ndarray
    class_names: list[str | None]


def _walk_path(data: Any, path: str) -> Any:
    current = data
    for chunk in path.split("."):
        if not chunk:
            continue
        if isinstance(current, list):
            current = current[int(chunk)]
        elif isinstance(current, dict):
            current = current.get(chunk)
        else:
            return None
    return current


def _coerce_to_list(data: Any) -> list[Any]:
    if data is None:
        return []
    if isinstance(data, list):
        return data
    return [data]


def _extract_polygons(raw_masks: Any) -> list[list[list[float]]]:
    polygons: list[list[list[float]]] = []
    if not isinstance(raw_masks, list):
        return polygons

    # Expected: list[polygon], polygon=list[[x,y], [x,y], ...]
    if raw_masks and isinstance(raw_masks[0], list) and raw_masks[0]:
        first = raw_masks[0][0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            polygons.append(raw_masks)  # type: ignore[arg-type]
            return polygons

    # Alternative: list[list[polygon]]
    for candidate in raw_masks:
        if not isinstance(candidate, list) or not candidate:
            continue
        if isinstance(candidate[0], (list, tuple)) and len(candidate[0]) == 2:
            polygons.append(candidate)  # type: ignore[arg-type]
    return polygons


def _polygons_to_mask(polygons: list[list[list[float]]], width: int, height: int) -> np.ndarray:
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        points = [(float(pt[0]), float(pt[1])) for pt in polygon]
        draw.polygon(points, outline=1, fill=1)
    return np.array(img, dtype=np.uint8)


class RoboflowWorkflowPredictor:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        workspace_name: str,
        workflow_id: str,
        predictions_path: str,
        class_name_field: str,
        use_cache: bool = True,
    ) -> None:
        try:
            from inference_sdk import InferenceHTTPClient
        except Exception as exc:
            raise ImportError(
                "Roboflow backend requires inference-sdk. Install with `pip install inference-sdk`."
            ) from exc

        self._client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self._workspace_name = workspace_name
        self._workflow_id = workflow_id
        self._predictions_path = predictions_path
        self._class_name_field = class_name_field
        self._use_cache = use_cache

    def _run_workflow(self, image_path: str, parameters: dict[str, Any] | None) -> Any:
        try:
            return self._client.run_workflow(
                workspace_name=self._workspace_name,
                workflow_id=self._workflow_id,
                images={"image": image_path},
                parameters=parameters or {},
                use_cache=self._use_cache,
            )
        except Exception as exc:
            raise RuntimeError(
                "Roboflow workflow call failed. "
                f"workflow={self._workspace_name}/{self._workflow_id}, "
                f"parameters={parameters or {}}. "
                "Check workflow required inputs and config backend.roboflow.* fields."
            ) from exc

    def predict(
        self, image_path: str, image_width: int, image_height: int, parameters: dict[str, Any] | None = None
    ) -> RoboflowPrediction:
        raw = self._run_workflow(image_path=image_path, parameters=parameters)
        root = raw
        if isinstance(root, list) and root:
            root = root[0]

        predictions_obj = _walk_path(root, self._predictions_path)
        if predictions_obj is None and isinstance(root, dict):
            # Fallback to common field if custom path is absent.
            predictions_obj = root.get("predictions")
        predictions = _coerce_to_list(predictions_obj)

        masks: list[np.ndarray] = []
        scores: list[float] = []
        class_names: list[str | None] = []

        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            polygons = _extract_polygons(pred.get("masks"))
            if not polygons:
                continue
            mask = _polygons_to_mask(polygons=polygons, width=image_width, height=image_height)
            masks.append(mask)
            scores.append(float(pred.get("confidence", pred.get("score", 1.0))))
            class_name = pred.get(self._class_name_field)
            class_names.append(str(class_name) if class_name is not None else None)

        if not masks:
            return RoboflowPrediction(
                masks=np.zeros((0, 0, 0), dtype=np.uint8),
                scores=np.array([], dtype=float),
                class_names=[],
            )
        return RoboflowPrediction(
            masks=np.stack(masks, axis=0),
            scores=np.array(scores, dtype=float),
            class_names=class_names,
        )
