#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage:"
  echo "  $0 <sam3_repo_path> <coco_json> <images_root> <dataset_root> <dataset_name> [num_gpus] [max_epochs]"
  echo
  echo "Example:"
  echo "  $0 /opt/sam3 \\"
  echo "     /workspace/sam-3/outputs/run-001/annotations/instances.json \\"
  echo "     /workspace/sam-3/data/images \\"
  echo "     /workspace/sam-3/data/sam3_train \\"
  echo "     my_dataset 1 20"
  exit 1
fi

SAM3_REPO="$1"
COCO_JSON="$2"
IMAGES_ROOT="$3"
DATASET_PARENT="$4"
DATASET_NAME="$5"
NUM_GPUS="${6:-1}"
MAX_EPOCHS="${7:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_ROOT="${DATASET_PARENT}/${DATASET_NAME}"
OUTPUT_CONFIG="${PROJECT_ROOT}/config/sam3_train_${DATASET_NAME}.yaml"
EXPERIMENT_LOG_DIR="${PROJECT_ROOT}/outputs/sam3_train_runs/${DATASET_NAME}"

echo "Step 1/3: Prepare train/valid/test dataset..."
python "${PROJECT_ROOT}/scripts/prepare_sam3_dataset.py" \
  --coco "${COCO_JSON}" \
  --images-root "${IMAGES_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 42

echo "Step 2/3: Generate SAM-3 train config..."
python "${PROJECT_ROOT}/scripts/generate_sam3_train_config.py" \
  --sam3-repo "${SAM3_REPO}" \
  --dataset-parent "${DATASET_PARENT}" \
  --dataset-name "${DATASET_NAME}" \
  --experiment-log-dir "${EXPERIMENT_LOG_DIR}" \
  --output-config "${OUTPUT_CONFIG}" \
  --num-gpus "${NUM_GPUS}" \
  --max-epochs "${MAX_EPOCHS}"

echo "Step 3/3: Start SAM-3 fine-tuning..."
bash "${PROJECT_ROOT}/scripts/run_sam3_train.sh" "${SAM3_REPO}" "${OUTPUT_CONFIG}" "${NUM_GPUS}"
