#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-data/apks}"
BELONG="${BELONG:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/static_reports}"
WORKERS="${WORKERS:-8}"

python -u phase1/static_feature_extract.py \
  --dataset-root "$DATASET_ROOT" \
  --belong "$BELONG" \
  --output "$OUTPUT_DIR" \
  --workers "$WORKERS"
