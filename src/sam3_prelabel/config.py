from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LabelPrompt:
    id: int
    name: str
    prompt: str


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    labels: list[LabelPrompt]
    score_threshold: float
    mask_threshold: float
    min_area: int
    max_masks_per_label: int
    polygon_approx_epsilon: float
    save_binary_masks: bool
    save_visualizations: bool
    image_extensions: tuple[str, ...]
    backend_name: str
    roboflow_api_url: str | None
    roboflow_api_key: str | None
    roboflow_workspace_name: str | None
    roboflow_workflow_id: str | None
    roboflow_use_cache: bool
    roboflow_per_label_requests: bool
    roboflow_prompt_parameter_name: str | None
    roboflow_predictions_path: str
    roboflow_class_name_field: str


def _require(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required config key: {key}")
    return data[key]


def load_config(config_path: str | Path) -> PipelineConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    paths = _require(raw, "paths")
    labels_raw = _require(raw, "labels")
    inference = raw.get("inference", {})
    output = raw.get("output", {})
    backend = raw.get("backend", {})
    roboflow = backend.get("roboflow", {})

    labels = [
        LabelPrompt(
            id=int(_require(item, "id")),
            name=str(_require(item, "name")),
            prompt=str(_require(item, "prompt")),
        )
        for item in labels_raw
    ]

    image_extensions = tuple(
        ext.lower()
        for ext in raw.get(
            "image_extensions",
            [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"],
        )
    )

    api_key = roboflow.get("api_key")
    api_key_env = roboflow.get("api_key_env")
    if api_key is None and api_key_env:
        api_key = os.getenv(str(api_key_env))

    return PipelineConfig(
        input_dir=Path(_require(paths, "input_dir")),
        output_dir=Path(_require(paths, "output_dir")),
        labels=labels,
        score_threshold=float(inference.get("score_threshold", 0.4)),
        mask_threshold=float(inference.get("mask_threshold", 0.5)),
        min_area=int(inference.get("min_area", 64)),
        max_masks_per_label=int(inference.get("max_masks_per_label", 20)),
        polygon_approx_epsilon=float(inference.get("polygon_approx_epsilon", 2.0)),
        save_binary_masks=bool(output.get("save_binary_masks", True)),
        save_visualizations=bool(output.get("save_visualizations", True)),
        image_extensions=image_extensions,
        backend_name=str(backend.get("name", "sam3")).lower(),
        roboflow_api_url=roboflow.get("api_url"),
        roboflow_api_key=api_key,
        roboflow_workspace_name=roboflow.get("workspace_name"),
        roboflow_workflow_id=roboflow.get("workflow_id"),
        roboflow_use_cache=bool(roboflow.get("use_cache", True)),
        roboflow_per_label_requests=bool(roboflow.get("per_label_requests", False)),
        roboflow_prompt_parameter_name=roboflow.get("prompt_parameter_name"),
        roboflow_predictions_path=str(roboflow.get("predictions_path", "predictions")),
        roboflow_class_name_field=str(roboflow.get("class_name_field", "class")),
    )
