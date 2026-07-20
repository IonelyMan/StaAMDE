#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-runs/image_convnextv2/$(date +%m-%d_%H-%M)}"
INPUT_DIR="${INPUT_DIR:-outputs/dex_images}"
ARCH="${ARCH:-convnextv2_atto}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-100}"
WORKERS="${WORKERS:-4}"

mkdir -p "$RUN_DIR"

python -u -m phase2.image.train \
  --input "$INPUT_DIR" \
  --output "$RUN_DIR" \
  --arch "$ARCH" \
  --image-size "$IMAGE_SIZE" "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --workers "$WORKERS" 2>&1 | tee "$RUN_DIR/train.log"
