#!/usr/bin/env bash
# Run fine-tuned SAM-3 polygon visualization + JSON sidecars on EC2 (or any Linux+CUDA host).
# Usage:
#   bash scripts/run_polygon_inference_check.sh
# Optional env overrides:
#   SAM3_REPO=~/sam3 SAM3_POC=~/sam3-poc CKPT=... IMG_DIR=... OUT_DIR=... SCORE=0.001 MAX_IMG=50
#   DEBUG=1          -> pass --debug (prints why boxes/scores are empty)
#   SIGMOID=1        -> pass --apply-sigmoid-to-scores (if raw scores are logits)
# Default IMG_DIR is **test**; train often returns zero boxes for this fine-tuned ckpt.

set -euo pipefail

SAM3_REPO="${SAM3_REPO:-$HOME/sam3}"
SAM3_POC="${SAM3_POC:-$HOME/sam3-poc}"
VENV_TRAIN="${VENV_TRAIN:-$SAM3_POC/.venv-train}"
CKPT="${CKPT:-$SAM3_POC/outputs/sam3_train_runs/my_dataset/checkpoints/model_only.pt}"
IMG_DIR="${IMG_DIR:-$SAM3_POC/data/sam3_train/my_dataset/test}"
OUT_DIR="${OUT_DIR:-$SAM3_POC/outputs/inference_polygon_vis_check_$(date +%Y%m%d_%H%M%S)}"
SCORE="${SCORE:-0.001}"
MAX_IMG="${MAX_IMG:-50}"
MAX_PER_LABEL="${MAX_PER_LABEL:-10}"

LABELS="${LABELS:-bottom_bun,cheese_slice,patty,tomato_slice,lettuce_leaf,top_bun,lettuce_container,tomato_container,cheese_rack,patty_rack,burger_box,onion_container,bun_container,onion_slice}"

DEBUG="${DEBUG:-0}"
SIGMOID="${SIGMOID:-0}"
extra_py=()
[[ "$DEBUG" == "1" ]] && extra_py+=(--debug)
[[ "$SIGMOID" == "1" ]] && extra_py+=(--apply-sigmoid-to-scores)

echo "SAM3_REPO=$SAM3_REPO"
echo "SAM3_POC=$SAM3_POC"
echo "OUT_DIR=$OUT_DIR"
echo "CKPT=$CKPT"
echo "IMG_DIR=$IMG_DIR"
echo "SCORE_THRESHOLD=$SCORE"
echo "DEBUG=$DEBUG SIGMOID=$SIGMOID"

test -d "$SAM3_REPO" || { echo "Missing SAM3 repo: $SAM3_REPO"; exit 1; }
test -d "$SAM3_POC" || { echo "Missing sam3-poc: $SAM3_POC"; exit 1; }
test -f "$CKPT" || { echo "Missing checkpoint: $CKPT"; exit 1; }
test -d "$IMG_DIR" || { echo "Missing image dir: $IMG_DIR"; exit 1; }
test -f "$VENV_TRAIN/bin/activate" || { echo "Missing venv: $VENV_TRAIN"; exit 1; }

mkdir -p "$OUT_DIR"

# shellcheck source=/dev/null
source "$VENV_TRAIN/bin/activate"
export PYTHONPATH="$SAM3_POC/src:${PYTHONPATH:-}"

cd "$SAM3_REPO"
python "$SAM3_POC/scripts/sam3_ft_inference_vis_polygons.py" \
  --ckpt "$CKPT" \
  --img-dir "$IMG_DIR" \
  --out-dir "$OUT_DIR" \
  --labels "$LABELS" \
  --max-images "$MAX_IMG" \
  --max-per-label "$MAX_PER_LABEL" \
  --score-threshold "$SCORE" \
  --rectangle-if-no-mask \
  --save-json \
  "${extra_py[@]}"

echo ""
echo "=== sample JSON (first file) ==="
first_json="$(ls "$OUT_DIR"/*_polygons.json 2>/dev/null | head -1 || true)"
if [[ -n "$first_json" ]]; then
  head -c 600 "$first_json"
  echo ""
  python3 - <<PY
import json, glob
import os
paths = sorted(glob.glob(os.path.join("$OUT_DIR", "*_polygons.json")))
nonempty = sum(1 for p in paths if json.load(open(p)))
print(f"json files: {len(paths)}, nonempty (has detections): {nonempty}")
PY
else
  echo "No *_polygons.json found under $OUT_DIR"
fi

echo ""
echo "Done. Images + JSON: $OUT_DIR"
