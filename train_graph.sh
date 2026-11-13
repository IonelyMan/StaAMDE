#!/usr/bin/env bash

LOGDIR="runs/graph/$(date +%m-%d_%H-%M-%S)"
LOG="$LOGDIR/train.log" # 月日+时分秒
mkdir -p "$LOGDIR"
INPUT_DIR="${GRAPH_INPUT_DIR:-/root/autodl-tmp/lzw/datasets/StaADME_dataset1/android_graphs}"

# 有效批次仍为 8；每次只把 1 张图送入 GPU，累积 8 次梯度后更新参数。
nohup python3 -u phase2/graph/train.py \
  --batch-size 8 \
  --micro-batch-size 1 \
  --activation-checkpointing \
  --input "$INPUT_DIR" \
  --model-type "atgatv2" \
  --workers 12 \
  --type-embedding-dim 16 \
  --info "ATGATv2：使用训练集类别权重；保留显存优化" \
  --output "$LOGDIR" > "$LOG" 2>&1 &

echo $!

