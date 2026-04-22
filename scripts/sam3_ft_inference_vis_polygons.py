#!/usr/bin/env python3
"""
Visualize fine-tuned SAM-3 image inference with **polygon outlines** (from masks).

Requires a Linux + CUDA machine with SAM-3 installed (``pip install -e <sam3-repo>``),
this repo on ``PYTHONPATH`` (or ``pip install -e .`` from repo root), and OpenCV.

Flow:
  1. ``build_sam3_image_model()`` then ``load_state_dict`` from your ``model_only.pt`` /
     full checkpoint (same pattern as ad-hoc EC2 inference).
  2. For each image and text label, ``Sam3Processor.set_text_prompt`` → ``masks``, ``boxes``, ``scores``.
  3. Threshold masks, convert each to polygons via ``binary_mask_to_polygons`` (same as pre-label COCO export).
  4. Draw polygons on the image (and optionally save a small JSON sidecar per image).

If the run returns **no masks** (e.g. fine-tune without segmentation head), use
``--rectangle-if-no-mask`` to draw the **bounding box as a 4-point polygon** so you still
get a closed polygon on the image.

``Sam3Processor`` defaults to ``confidence_threshold=0.5`` and drops almost all
fine-tuned scores before ``boxes`` are returned. Use ``--processor-confidence-threshold``
(we default to ``0.001``). Fine-tuned heads often yield max combined scores well
below ``0.01``; if JSON is still empty, try ``1e-4`` or ``0.0``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from sam3_prelabel.coco_exporter import binary_mask_to_polygons  # noqa: E402


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


def _xyxy_to_polygon_flat(box: np.ndarray | list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _draw_polygons(
    draw: ImageDraw.ImageDraw,
    polygons: list[list[float]],
    outline: tuple[int, int, int],
    width: int = 2,
) -> None:
    for poly in polygons:
        if len(poly) < 6:
            continue
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        draw.polygon(pts, outline=outline, width=width)


def _mask_index_to_binary(
    masks_np: np.ndarray | None, idx: int, mask_threshold: float
) -> np.ndarray | None:
    if masks_np is None or masks_np.size == 0:
        return None
    if masks_np.ndim == 4:
        masks_np = np.squeeze(masks_np, axis=1)
    if masks_np.ndim != 3 or idx >= masks_np.shape[0]:
        return None
    return _normalize_masks(masks_np[idx : idx + 1], mask_threshold)


def _tensor_to_numpy(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):
        tt = t.detach()
        # NumPy cannot ingest torch.bfloat16 directly; cast floating tensors.
        if hasattr(tt, "is_floating_point") and tt.is_floating_point():
            tt = tt.float()
        return tt.cpu().numpy()
    return np.array(t)


def _polygons_for_instance(
    masks_np: np.ndarray | None,
    idx: int,
    box_row: np.ndarray,
    mask_threshold: float,
    polygon_epsilon: float,
    rectangle_if_no_mask: bool,
) -> list[list[float]]:
    polys: list[list[float]] = []
    mb = _mask_index_to_binary(masks_np, int(idx), mask_threshold)
    if mb is not None and mb.shape[0] >= 1:
        polys = binary_mask_to_polygons(mb[0], epsilon=polygon_epsilon)
    if not polys and rectangle_if_no_mask:
        polys = [_xyxy_to_polygon_flat(box_row)]
    return polys


def _append_label_predictions(
    draw: ImageDraw.ImageDraw,
    sidecar: list[dict],
    label: str,
    color: tuple[int, int, int],
    prompt_out: dict[str, Any],
    args: argparse.Namespace,
    image_name: str = "",
) -> None:
    boxes = prompt_out.get("boxes")
    scores = prompt_out.get("scores")
    masks_raw = prompt_out.get("masks")
    if boxes is None:
        if getattr(args, "debug", False):
            print(
                f"DEBUG {image_name} label={label!r}: boxes is None (no detection tensor)",
                flush=True,
            )
        return
    n_box = len(boxes)
    if n_box == 0:
        if getattr(args, "debug", False):
            shape = getattr(boxes, "shape", "?")
            print(
                f"DEBUG {image_name} label={label!r}: 0 proposals "
                f"(empty detections; tensor shape={shape})",
                flush=True,
            )
        return
    boxes_np = _tensor_to_numpy(boxes)
    scores_np = _tensor_to_numpy(scores)
    masks_np = _tensor_to_numpy(masks_raw) if masks_raw is not None else None

    if getattr(args, "apply_sigmoid_to_scores", False):
        scores_np = 1.0 / (1.0 + np.exp(-np.clip(scores_np, -50.0, 50.0)))

    if getattr(args, "debug", False):
        print(
            f"DEBUG {image_name} label={label!r}: n_boxes={boxes_np.shape[0]} "
            f"score_min={float(scores_np.min()):.4g} score_max={float(scores_np.max()):.4g}",
            flush=True,
        )

    for idx in np.argsort(-scores_np)[: args.max_per_label]:
        s = float(scores_np[idx])
        if s < args.score_threshold:
            continue
        polys = _polygons_for_instance(
            masks_np,
            int(idx),
            boxes_np[idx],
            args.mask_threshold,
            args.polygon_epsilon,
            args.rectangle_if_no_mask,
        )
        if not polys:
            continue
        _draw_polygons(draw, polys, outline=color, width=2)
        x1, y1, *_ = [float(t) for t in boxes_np[idx].tolist()]
        draw.text((x1 + 2, y1 + 2), f"{label} {s:.2f}", fill=color)
        if args.save_json:
            sidecar.append(
                {
                    "label": label,
                    "score": s,
                    "bbox_xyxy": [float(t) for t in boxes_np[idx].tolist()],
                    "segmentation": polys,
                }
            )


def _render_image_polygons(
    processor: Any,
    path: Path,
    label_list: list[str],
    colors: list[tuple[int, int, int]],
    args: argparse.Namespace,
) -> tuple[Image.Image, list[dict]]:
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    sidecar: list[dict] = []

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state_img = processor.set_image(img)
        for li, label in enumerate(label_list):
            prompt_out = processor.set_text_prompt(state=state_img, prompt=label)
            color = colors[li % len(colors)]
            _append_label_predictions(
                draw, sidecar, label, color, prompt_out, args, image_name=path.name
            )

    return img, sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="Fine-tuned weights (.pt)")
    parser.add_argument("--img-dir", type=Path, required=True, help="Directory of input images")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for rendered images")
    parser.add_argument(
        "--labels",
        type=str,
        default="bottom_bun,cheese_slice,patty,tomato_slice,lettuce_leaf,top_bun",
        help="Comma-separated text prompts (same vocabulary as training when possible)",
    )
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--max-per-label", type=int, default=3, help="Top-K instances per label by score")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--polygon-epsilon", type=float, default=2.0)
    parser.add_argument(
        "--rectangle-if-no-mask",
        action="store_true",
        help="If masks are empty, draw predicted box as a rectangle polygon instead",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Write one <stem>_polygons.json per image (always; empty [] if no hits)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-label box counts and score min/max (why JSON can be [])",
    )
    parser.add_argument(
        "--apply-sigmoid-to-scores",
        action="store_true",
        help="If model returns logits, map scores with sigmoid before thresholding/sorting",
    )
    parser.add_argument(
        "--processor-confidence-threshold",
        type=float,
        default=0.001,
        help=(
            "Sam3Processor keeps only predictions with score > this value before returning "
            "boxes (SAM-3 default is 0.5). Fine-tuned runs often peak around 1e-3–1e-2; "
            "values like 0.05 can drop every proposal."
        ),
    )
    args = parser.parse_args()

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("load_state_dict missing:", len(missing), "unexpected:", len(unexpected))

    model = model.to("cuda").eval()
    processor = Sam3Processor(
        model, confidence_threshold=args.processor_confidence_threshold
    )
    print(
        "Sam3Processor confidence_threshold:",
        args.processor_confidence_threshold,
        flush=True,
    )

    label_list = [s.strip() for s in args.labels.split(",") if s.strip()]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted([p for p in args.img_dir.iterdir() if p.suffix.lower() in exts])[
        : args.max_images
    ]
    print(f"Processing {len(paths)} images, {len(label_list)} labels each")

    colors = [
        (255, 80, 80),
        (80, 200, 120),
        (80, 140, 255),
        (255, 200, 80),
        (200, 80, 255),
        (80, 220, 220),
        (220, 120, 80),
        (180, 180, 255),
    ]

    for path in paths:
        img, sidecar = _render_image_polygons(
            processor, path, label_list, colors, args
        )

        out_path = args.out_dir / path.name
        img.save(out_path)
        print("wrote", out_path)
        if args.save_json:
            jp = out_path.with_name(out_path.stem + "_polygons.json")
            jp.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
            print("  json:", jp, "count:", len(sidecar))


if __name__ == "__main__":
    main()
