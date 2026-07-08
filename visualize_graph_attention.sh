#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-runs/hetero_gatv2}"
SAMPLE_ID="${SAMPLE_ID:-}"
TOP_K_NODES="${TOP_K_NODES:-30}"
NEIGHBOR_HOPS="${NEIGHBOR_HOPS:-1}"
MAX_DISPLAY_NODES="${MAX_DISPLAY_NODES:-60}"
MAX_DISPLAY_EDGES="${MAX_DISPLAY_EDGES:-180}"
LAYOUT="${LAYOUT:-circular}"

ARGS=(
  -m phase2.graph.visualize_attention
  --run-dir "$RUN_DIR"
  --top-k-nodes "$TOP_K_NODES"
  --neighbor-hops "$NEIGHBOR_HOPS"
  --max-display-nodes "$MAX_DISPLAY_NODES"
  --max-display-edges "$MAX_DISPLAY_EDGES"
  --layout "$LAYOUT"
)

if [[ -n "$SAMPLE_ID" ]]; then
  ARGS+=(--sample-id "$SAMPLE_ID")
fi

python "${ARGS[@]}"
