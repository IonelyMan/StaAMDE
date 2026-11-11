#!/usr/bin/env bash


INPUT_DIR="${GRAPH_INPUT_DIR:-/home/linux/7T/lzw/datasets/android_zoo/EEMAD_datasets/android_graphs}"
MODEL_PATH="${GRAPH_MODEL_PATH:-runs/graph/07-03_15-37_best/models/gatv2_model.pt}"
SPLIT="${GRAPH_SPLIT:-val}"
mkdir -p inference/atgatv2
nohup python3 -m phase2.graph.inference \
  --input "$INPUT_DIR" \
  --model-path "$MODEL_PATH" \
  --output inference/atgatv2 \
  --split "$SPLIT" \
  --workers 20 > inference/atgatv2/graph.log 2>&1 &
