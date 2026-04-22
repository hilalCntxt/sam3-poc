#!/usr/bin/env bash
# Run fine-tuned SAM-3 polygon visualization + JSON sidecars on EC2 (or any Linux+CUDA host).
# Usage:
#   bash scripts/run_polygon_inference_check.sh
# Optional env overrides:
#   SAM3_REPO=~/sam3 SAM3_POC=~/sam3-poc CKPT=... IMG_DIR=... OUT_DIR=... SCORE=0.005 MAX_IMG=50
#   DEBUG=1          -> pass --debug (prints why boxes/scores are empty)
#   SIGMOID=1        -> pass --apply-sigmoid-to-scores (if raw scores are logits)
#   PROC_THR=0.001   -> Sam3Processor internal filter (FT runs often peak ~0.002; 0.05 drops all)
#   AUTO_FIX_PROC_THR=0 -> keep PROC_THR as-is even if suspiciously high
# Default IMG_DIR is **test**; train often returns zero boxes for this fine-tuned ckpt.

set -euo pipefail

SAM3_REPO="${SAM3_REPO:-$HOME/sam3}"
SAM3_POC="${SAM3_POC:-$HOME/sam3-poc}"
VENV_TRAIN="${VENV_TRAIN:-$SAM3_POC/.venv-train}"
CKPT="${CKPT:-$SAM3_POC/outputs/sam3_train_runs/my_dataset/checkpoints/model_only.pt}"
IMG_DIR="${IMG_DIR:-$SAM3_POC/data/sam3_train/my_dataset/test}"
OUT_DIR="${OUT_DIR:-$SAM3_POC/outputs/inference_polygon_vis_check_$(date +%Y%m%d_%H%M%S)}"
SCORE="${SCORE:-0.005}"
MAX_IMG="${MAX_IMG:-50}"
MAX_PER_LABEL="${MAX_PER_LABEL:-3}"
NMS_IOU="${NMS_IOU:-0.5}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-$SAM3_POC/config/pipeline.yaml}"

LABELS_DEFAULT="bottom_bun,cheese_slice,patty,tomato_slice,lettuce_leaf,top_bun,lettuce_container,tomato_container,cheese_rack,patty_rack,burger_box,onion_container,bun_container,onion_slice"
LABELS_INPUT="${LABELS:-}"
if [[ -f "$PIPELINE_CONFIG" ]]; then
  LABELS="$(python3 - "$PIPELINE_CONFIG" "$LABELS_INPUT" <<'PY'
import sys
import yaml

cfg_path = sys.argv[1]
raw_input = sys.argv[2].strip()
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
labels = cfg.get("labels", [])
allowed = [str(item.get("prompt", "")).strip() for item in labels if str(item.get("prompt", "")).strip()]
allowed_set = set(allowed)

if raw_input:
    requested = [s.strip() for s in raw_input.split(",") if s.strip()]
    filtered = [s for s in requested if s in allowed_set]
    print(",".join(filtered))
else:
    print(",".join(allowed))
PY
)"
else
  LABELS="${LABELS_INPUT:-$LABELS_DEFAULT}"
fi
if [[ -z "$LABELS" ]]; then
  echo "No valid labels after filtering. Check PIPELINE_CONFIG=$PIPELINE_CONFIG or LABELS."
  exit 1
fi

DEBUG="${DEBUG:-0}"
SIGMOID="${SIGMOID:-0}"
PROC_THR="${PROC_THR:-0.001}"
AUTO_FIX_PROC_THR="${AUTO_FIX_PROC_THR:-1}"

# Many shells keep exported PROC_THR from old sessions (often 0.05), which
# filters out all fine-tuned detections before boxes are returned.
if python3 - "$PROC_THR" <<'PY'
import sys
thr = float(sys.argv[1])
sys.exit(0 if thr > 0.01 else 1)
PY
then
  if [[ "$AUTO_FIX_PROC_THR" == "1" ]]; then
    echo "WARNING: PROC_THR=$PROC_THR is likely too high for this fine-tuned checkpoint."
    echo "         Auto-fixing PROC_THR to 0.001 (set AUTO_FIX_PROC_THR=0 to disable)."
    PROC_THR="0.001"
  else
    echo "WARNING: PROC_THR=$PROC_THR is high; this can produce zero proposals."
  fi
fi

extra_py=(--processor-confidence-threshold "$PROC_THR")
[[ "$DEBUG" == "1" ]] && extra_py+=(--debug)
[[ "$SIGMOID" == "1" ]] && extra_py+=(--apply-sigmoid-to-scores)

echo "SAM3_REPO=$SAM3_REPO"
echo "SAM3_POC=$SAM3_POC"
echo "OUT_DIR=$OUT_DIR"
echo "CKPT=$CKPT"
echo "IMG_DIR=$IMG_DIR"
echo "SCORE_THRESHOLD=$SCORE"
echo "NMS_IOU=$NMS_IOU"
echo "PIPELINE_CONFIG=$PIPELINE_CONFIG"
echo "LABELS=$LABELS"
echo "DEBUG=$DEBUG SIGMOID=$SIGMOID PROC_THR=$PROC_THR"

if [[ "$IMG_DIR" == *"/train" ]] || [[ "$IMG_DIR" == *"/train/" ]]; then
  echo "WARNING: IMG_DIR points at TRAIN (${IMG_DIR})."
  echo "         This checkpoint often returns 0 proposals on train; use .../test or unset IMG_DIR."
fi

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
  --nms-iou-threshold "$NMS_IOU" \
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
