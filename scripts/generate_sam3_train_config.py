#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a SAM-3 train config for your custom dataset.")
    parser.add_argument("--sam3-repo", required=True, help="Path to cloned facebookresearch/sam3 repo.")
    parser.add_argument(
        "--template-config",
        default="sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml",
        help="Template config path relative to --sam3-repo.",
    )
    parser.add_argument("--dataset-parent", required=True, help="Root containing dataset folder.")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset folder under dataset-parent that contains train/valid/test.",
    )
    parser.add_argument("--experiment-log-dir", required=True, help="Directory for training logs/checkpoints.")
    parser.add_argument("--output-config", required=True, help="Where to write generated YAML config.")
    parser.add_argument("--num-gpus", type=int, default=1, help="GPUs per node for launcher config.")
    parser.add_argument("--max-epochs", type=int, default=20, help="Maximum epochs.")
    args = parser.parse_args()

    sam3_repo = Path(args.sam3_repo)
    template_path = sam3_repo / args.template_config
    config = _read_yaml(template_path)

    config.setdefault("paths", {})
    config["paths"]["roboflow_vl_100_root"] = str(Path(args.dataset_parent))
    config["paths"]["experiment_log_dir"] = str(Path(args.experiment_log_dir))
    config["paths"]["bpe_path"] = str(sam3_repo / "assets/bpe_simple_vocab_16e6.txt.gz")

    config.setdefault("roboflow_train", {})
    config["roboflow_train"]["supercategory"] = args.dataset_name
    config["roboflow_train"]["num_images"] = None  # use all available images

    config.setdefault("trainer", {})
    config["trainer"]["max_epochs"] = int(args.max_epochs)
    config["trainer"]["skip_saving_ckpts"] = False

    config.setdefault("launcher", {})
    config["launcher"]["gpus_per_node"] = int(args.num_gpus)

    config.setdefault("submitit", {})
    config["submitit"]["use_cluster"] = False
    if "job_array" in config["submitit"]:
        config["submitit"]["job_array"]["num_tasks"] = 1
        config["submitit"]["job_array"]["task_index"] = 0

    config["all_roboflow_supercategories"] = [args.dataset_name]

    _write_yaml(Path(args.output_config), config)
    print(f"Generated SAM-3 train config: {args.output_config}")


if __name__ == "__main__":
    main()
