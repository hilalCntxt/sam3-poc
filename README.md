# Image Pre-Labeling Pipeline (SAM-3 or Roboflow Workflow)

This project provides a full pre-labeling pipeline for large image batches with two backends:

- `sam3`: local [SAM-3](https://github.com/facebookresearch/sam3) inference
- `roboflow`: hosted workflow inference via `inference_sdk`

- Text-prompt based class pre-labeling
- Batch image processing from a folder
- Mask filtering by confidence/area
- COCO instance segmentation export (`instances.json`)
- Optional binary masks and visualization previews

## 1) Environment Setup

Use Python `3.12+`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## 2) Prepare Data

Put your images under:

```text
data/images/
```

You can use nested folders. The pipeline scans recursively.

## 3) Configure Backend + Labels

Copy and edit:

```bash
cp config/pipeline.example.yaml config/pipeline.yaml
```

Set:

- `paths.input_dir`
- `paths.output_dir`
- `backend.name` (`sam3` or `roboflow`)
- `labels` (`id`, `name`, `prompt`)
- filtering and export options in `inference` and `output`

### Roboflow backend setup

Use this in `config/pipeline.yaml`:

- `backend.name: roboflow`
- `backend.roboflow.api_url`
- `backend.roboflow.workspace_name`
- `backend.roboflow.workflow_id`
- `backend.roboflow.api_key_env` (recommended)

Then export your API key:

```bash
export ROBOFLOW_API_KEY="your_api_key"
```

If your workflow returns predictions in a nested object, adjust:
- `backend.roboflow.predictions_path`

If your workflow class field is not `class`, adjust:
- `backend.roboflow.class_name_field`

### SAM-3 backend setup

```bash
git clone https://github.com/facebookresearch/sam3.git
pip install -e ./sam3
hf auth login
```

Note: SAM-3 backend requires Linux + NVIDIA CUDA runtime, and is best used in a separate virtual environment from Roboflow due dependency differences.

## 4) Run Pre-Labeling

```bash
sam3-prelabel --config config/pipeline.yaml
```

or:

```bash
python -m sam3_prelabel.cli --config config/pipeline.yaml
```

## 5) Output Structure

The pipeline writes:

```text
outputs/run-001/
  annotations/instances.json
  run_stats.json
  masks/                 # optional
  visualizations/        # optional
```

- `annotations/instances.json`: COCO-format instance segmentation annotations
- `run_stats.json`: run summary and skip counts
- `masks/`: one PNG per predicted instance mask
- `visualizations/`: overlay previews for quick QA

## 6) Suggested Annotation Workflow

1. Import `instances.json` to your annotation platform (CVAT/Label Studio/other COCO-compatible tools).
2. Review only uncertain or noisy labels first (based on low confidence classes).
3. Correct masks and export clean labels for training.
4. Re-run this pipeline with improved prompts (`labels[].prompt`) to iteratively improve quality.

## 7) Fine-Tune SAM-3 on GPU

Use a Linux + NVIDIA CUDA machine.

### A) Prepare a SAM-3 training environment

```bash
conda create -n sam3-ft python=3.12 -y
conda activate sam3-ft
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e ".[train]"
```

### B) Convert pre-label COCO output to SAM-3 dataset layout

From this repo root:

```bash
python scripts/prepare_sam3_dataset.py \
  --coco outputs/run-001/annotations/instances.json \
  --images-root data/images \
  --dataset-root data/sam3_train/my_dataset \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 42
```

This creates:

```text
data/sam3_train/my_dataset/
  train/_annotations.coco.json
  valid/_annotations.coco.json
  test/_annotations.coco.json
  ...images...
```

### C) Generate SAM-3 train config

```bash
python scripts/generate_sam3_train_config.py \
  --sam3-repo /path/to/sam3 \
  --dataset-parent /absolute/path/to/sam-3/data/sam3_train \
  --dataset-name my_dataset \
  --experiment-log-dir /absolute/path/to/sam-3/outputs/sam3_train_runs/run-001 \
  --output-config /absolute/path/to/sam-3/config/sam3_train.yaml \
  --num-gpus 1 \
  --max-epochs 20
```

### D) Start fine-tuning

```bash
bash scripts/run_sam3_train.sh /path/to/sam3 /absolute/path/to/sam-3/config/sam3_train.yaml 1
```

For multi-GPU on one machine, set the last value to `2`, `4`, etc.

### E) One-command launcher (all steps)

Runs dataset split + config generation + training in one command:

```bash
bash scripts/run_all_sam3_finetune.sh \
  /path/to/sam3 \
  /absolute/path/to/sam-3/outputs/run-001/annotations/instances.json \
  /absolute/path/to/sam-3/data/images \
  /absolute/path/to/sam-3/data/sam3_train \
  my_dataset \
  1 \
  20
```

## Troubleshooting

- **`SAM-3 import failed`**  
  Make sure `pip install -e ./sam3` is done in the active environment.

- **`ModuleNotFoundError: triton` on macOS**  
  Use `backend.name: roboflow`, or run the SAM-3 backend on Linux + NVIDIA CUDA.

- **No masks for a label**  
  Try a more specific prompt (example: `"construction worker helmet"` instead of `"helmet"`).

- **Too many noisy masks**  
  Increase `score_threshold`, increase `min_area`, or reduce `max_masks_per_label`.
