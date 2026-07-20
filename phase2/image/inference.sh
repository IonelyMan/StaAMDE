#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-runs/image_convnextv2}"
INPUT_DIR="${INPUT_DIR:-outputs/dex_images}"
MODEL_PATH="${MODEL_PATH:-$RUN_DIR/models/best_model.pt}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
WORKERS="${WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"

python -u -m phase2.image.inference \
  --input "$INPUT_DIR" \
  --model-path "$MODEL_PATH" \
  --output "$RUN_DIR" \
  --split test \
  --image-size "$IMAGE_SIZE" "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS"
