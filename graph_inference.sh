#!/usr/bin/env bash


INPUT_DIR="${GRAPH_INPUT_DIR:-/root/autodl-tmp/lzw/datasets/StaADME_dataset1/android_graphs}"
MODEL_PATH="${GRAPH_MODEL_PATH:-runs/graph/11-13_00-28-48/models/atgatv2_model.pt}"
SPLIT="${GRAPH_SPLIT:-val}"
mkdir -p inference/atgatv2
nohup python3 -m phase2.graph.inference \
  --input "$INPUT_DIR" \
  --model-path "$MODEL_PATH" \
  --output inference/atgatv2 \
  --split "$SPLIT" \
  --workers 20 > inference/atgatv2/graph.log 2>&1 &
