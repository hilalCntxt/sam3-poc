#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _split_image_ids(
    images: list[dict[str, Any]], train_ratio: float, val_ratio: float, seed: int
) -> tuple[set[int], set[int], set[int]]:
    image_ids = [int(img["id"]) for img in images]
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train : n_train + n_val])
    test_ids = set(image_ids[n_train + n_val : n_train + n_val + n_test])
    return train_ids, val_ids, test_ids


def _subset_coco(
    coco: dict[str, Any], image_ids: set[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    images = [img for img in coco.get("images", []) if int(img["id"]) in image_ids]
    anns = [ann for ann in coco.get("annotations", []) if int(ann["image_id"]) in image_ids]

    # Re-number annotation IDs for clean train/val/test files.
    for idx, ann in enumerate(anns, start=1):
        ann["id"] = idx
    return images, anns


def _copy_images(
    images: list[dict[str, Any]],
    input_images_root: Path,
    split_dir: Path,
) -> None:
    for image in images:
        rel = Path(str(image["file_name"]))
        src = input_images_root / rel
        dst = split_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_split(
    coco: dict[str, Any],
    image_ids: set[int],
    input_images_root: Path,
    dataset_root: Path,
    split_name: str,
) -> None:
    split_dir = dataset_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    split_images, split_annotations = _subset_coco(coco, image_ids=image_ids)
    _copy_images(split_images, input_images_root=input_images_root, split_dir=split_dir)

    split_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": split_images,
        "annotations": split_annotations,
        "categories": coco.get("categories", []),
    }
    _save_json(split_dir / "_annotations.coco.json", split_coco)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split COCO pre-label output into SAM-3 train/valid/test dataset layout."
    )
    parser.add_argument(
        "--coco",
        required=True,
        help="Path to source COCO instances.json (e.g., outputs/run-001/annotations/instances.json).",
    )
    parser.add_argument(
        "--images-root",
        required=True,
        help="Root directory corresponding to COCO file_name paths (e.g., data/images).",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Output root for SAM-3 dataset (contains train/valid/test).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("train-ratio and val-ratio must be > 0.")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train-ratio + val-ratio must be < 1.0.")

    coco_path = Path(args.coco)
    images_root = Path(args.images_root)
    dataset_root = Path(args.dataset_root)

    coco = _load_json(coco_path)
    images = coco.get("images", [])
    if not images:
        raise RuntimeError("No images found in COCO file.")

    train_ids, val_ids, test_ids = _split_image_ids(
        images=images, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    _write_split(coco, train_ids, images_root, dataset_root, "train")
    _write_split(coco, val_ids, images_root, dataset_root, "valid")
    _write_split(coco, test_ids, images_root, dataset_root, "test")

    print("SAM-3 dataset prepared.")
    print(f"train images: {len(train_ids)}")
    print(f"valid images: {len(val_ids)}")
    print(f"test images: {len(test_ids)}")
    print(f"dataset root: {dataset_root}")


if __name__ == "__main__":
    main()
