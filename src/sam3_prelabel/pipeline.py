from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .coco_exporter import (
    CocoAnnotation,
    CocoImage,
    binary_mask_to_polygons,
    ensure_output_dirs,
    mask_to_bbox,
    overlay_mask,
)
from .config import PipelineConfig
from .roboflow_runner import RoboflowWorkflowPredictor
from .sam3_runner import Sam3ImagePredictor


@dataclass
class PipelineStats:
    total_images: int = 0
    total_annotations: int = 0
    skipped_empty_masks: int = 0
    skipped_low_score: int = 0
    skipped_small_area: int = 0


def _list_images(input_dir: Path, exts: tuple[str, ...]) -> list[Path]:
    paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    return sorted(paths)


def _normalize_masks(masks: np.ndarray, mask_threshold: float) -> np.ndarray:
    if masks.size == 0:
        return np.zeros((0, 0, 0), dtype=np.uint8)

    if masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim > 3:
        masks = np.squeeze(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]

    return (masks > mask_threshold).astype(np.uint8)


def _normalize_label_key(value: str) -> str:
    return value.strip().lower()


def _build_roboflow_label_parameters(
    config: PipelineConfig, prompts: list[str], single_prompt: str | None = None
) -> dict[str, Any]:
    """
    Build workflow parameters for Roboflow blocks that expect class inputs.

    Default behavior:
    - batch mode (`single_prompt=None`): {"classes": [<all prompts>]}
    - per-label mode (`single_prompt=<prompt>`): {"classes": "<prompt>"}
    """
    if config.roboflow_prompt_parameter_name:
        param_name = config.roboflow_prompt_parameter_name
    else:
        # Common input name used by SAM/SAM3 workflow blocks.
        param_name = "classes"

    if single_prompt is None:
        return {param_name: prompts}
    return {param_name: single_prompt}


def run_pipeline(config: PipelineConfig) -> PipelineStats:
    if not config.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")

    sam3_predictor: Sam3ImagePredictor | None = None
    rf_predictor: RoboflowWorkflowPredictor | None = None
    if config.backend_name == "sam3":
        sam3_predictor = Sam3ImagePredictor()
    elif config.backend_name == "roboflow":
        missing = []
        if not config.roboflow_api_url:
            missing.append("backend.roboflow.api_url")
        if not config.roboflow_api_key:
            missing.append("backend.roboflow.api_key or api_key_env")
        if not config.roboflow_workspace_name:
            missing.append("backend.roboflow.workspace_name")
        if not config.roboflow_workflow_id:
            missing.append("backend.roboflow.workflow_id")
        if missing:
            raise ValueError(f"Roboflow backend is missing required settings: {', '.join(missing)}")
        rf_predictor = RoboflowWorkflowPredictor(
            api_url=config.roboflow_api_url,
            api_key=config.roboflow_api_key,
            workspace_name=config.roboflow_workspace_name,
            workflow_id=config.roboflow_workflow_id,
            predictions_path=config.roboflow_predictions_path,
            class_name_field=config.roboflow_class_name_field,
            use_cache=config.roboflow_use_cache,
        )
    else:
        raise ValueError("Unsupported backend name. Use either 'sam3' or 'roboflow'.")
    out_dirs = ensure_output_dirs(
        config.output_dir,
        save_masks=config.save_binary_masks,
        save_visualizations=config.save_visualizations,
    )

    image_paths = _list_images(config.input_dir, config.image_extensions)
    if not image_paths:
        raise RuntimeError(f"No images found under {config.input_dir}")

    categories = [{"id": label.id, "name": label.name, "supercategory": "object"} for label in config.labels]
    label_map = {_normalize_label_key(label.name): label for label in config.labels}
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    stats = PipelineStats(total_images=len(image_paths))

    ann_id = 1
    image_id = 1

    color_map = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 80, 255),
        (255, 220, 80),
        (180, 80, 255),
        (80, 220, 255),
    ]

    for image_path in tqdm(image_paths, desc="Pre-labeling images"):
        pil_img = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        vis_img = img_bgr.copy()
        state = sam3_predictor.set_image(pil_img) if sam3_predictor else None

        height, width = img_np.shape[:2]
        coco_images.append(
            CocoImage(
                id=image_id,
                file_name=str(image_path.relative_to(config.input_dir)),
                width=width,
                height=height,
            ).to_dict()
        )

        prediction_batches: list[tuple[int, str, np.ndarray, np.ndarray]]
        prediction_batches = []

        if sam3_predictor is not None and state is not None:
            for label in config.labels:
                pred = sam3_predictor.predict_from_state(state=state, prompt=label.prompt)
                prediction_batches.append((label.id, label.name, pred.masks, pred.scores.reshape(-1)))
        elif rf_predictor is not None:
            if config.roboflow_per_label_requests:
                for label in config.labels:
                    params = _build_roboflow_label_parameters(
                        config=config,
                        prompts=[entry.prompt for entry in config.labels],
                        single_prompt=label.prompt,
                    )
                    pred = rf_predictor.predict(
                        image_path=str(image_path),
                        image_width=width,
                        image_height=height,
                        parameters=params,
                    )
                    prediction_batches.append((label.id, label.name, pred.masks, pred.scores))
            else:
                params = _build_roboflow_label_parameters(
                    config=config,
                    prompts=[entry.prompt for entry in config.labels],
                )
                pred = rf_predictor.predict(
                    image_path=str(image_path),
                    image_width=width,
                    image_height=height,
                    parameters=params,
                )
                for idx, class_name in enumerate(pred.class_names):
                    if class_name is None:
                        continue
                    label = label_map.get(_normalize_label_key(class_name))
                    if label is None:
                        continue
                    prediction_batches.append(
                        (
                            label.id,
                            label.name,
                            pred.masks[idx : idx + 1],
                            pred.scores[idx : idx + 1],
                        )
                    )

        for category_id, category_name, raw_masks, raw_scores in prediction_batches:
            masks = _normalize_masks(raw_masks, config.mask_threshold)
            scores = raw_scores.reshape(-1) if raw_scores.size else np.array([])

            if masks.size == 0:
                continue

            for idx in range(min(len(masks), config.max_masks_per_label)):
                score = float(scores[idx]) if idx < len(scores) else 1.0
                if score < config.score_threshold:
                    stats.skipped_low_score += 1
                    continue

                mask = masks[idx]
                area = int(mask.sum())
                if area == 0:
                    stats.skipped_empty_masks += 1
                    continue
                if area < config.min_area:
                    stats.skipped_small_area += 1
                    continue

                polygons = binary_mask_to_polygons(mask, epsilon=config.polygon_approx_epsilon)
                if not polygons:
                    continue

                bbox = mask_to_bbox(mask)
                coco_annotations.append(
                    CocoAnnotation(
                        id=ann_id,
                        image_id=image_id,
                        category_id=category_id,
                        segmentation=polygons,
                        area=float(area),
                        bbox=bbox,
                    ).to_dict()
                )
                ann_id += 1
                stats.total_annotations += 1

                if config.save_binary_masks:
                    stem = image_path.stem
                    mask_name = f"{stem}__{category_name}__{idx:03d}.png"
                    cv2.imwrite(str(out_dirs["masks"] / mask_name), (mask * 255).astype(np.uint8))

                if config.save_visualizations:
                    color = color_map[(category_id - 1) % len(color_map)]
                    vis_img = overlay_mask(vis_img, mask, color)

        if config.save_visualizations:
            vis_name = f"{image_path.stem}__preview.jpg"
            cv2.imwrite(str(out_dirs["visualizations"] / vis_name), vis_img)

        image_id += 1

    coco = {
        "info": {
            "description": "SAM-3 pre-labeling output",
            "version": "1.0",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    with (out_dirs["annotations"] / "instances.json").open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    with (out_dirs["base"] / "run_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats.__dict__, f, ensure_ascii=False, indent=2)

    return stats
