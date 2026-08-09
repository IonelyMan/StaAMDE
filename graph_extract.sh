#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-YOUR_APKS_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hetero_graphs}"

python -u phase1/build_heterogeneous.py \
  --dataset-root "$DATASET_ROOT" \
  --belong "test" \
  --output "$OUTPUT_DIR" \
  --workers 8
