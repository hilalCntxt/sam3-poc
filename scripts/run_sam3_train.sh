#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <sam3_repo_path> <config_yaml> [num_gpus]"
  exit 1
fi

SAM3_REPO="$1"
CONFIG_YAML="$2"
NUM_GPUS="${3:-1}"

python "$SAM3_REPO/sam3/train/train.py" \
  -c "$CONFIG_YAML" \
  --use-cluster 0 \
  --num-gpus "$NUM_GPUS"
