#!/usr/bin/env bash

LOGDIR="runs/graph/$(date +%m-%d_%H-%M-%S)"
LOG="$LOGDIR/train.log" # 月日+时分秒
mkdir -p "$LOGDIR"
INPUT_DIR="${GRAPH_INPUT_DIR:-/home/linux/7T/lzw/datasets/android_zoo/android_graphs}"

# 有效批次仍为 8；每次只把 1 张图送入 GPU，累积 8 次梯度后更新参数。
nohup python3 -u ./phase2/graph/train.py \
  --batch-size 8 \
  --micro-batch-size 1 \
  --activation-checkpointing \
  --input "$INPUT_DIR" \
  --model-type "atgatv2" \
  --workers 12 \
  --type-embedding-dim 16 \
  --readout mean_max \
  --class-weight sqrt \
  --lr-scheduler plateau \
  --grad-clip 2 \
  --patience 18 \
  --info "ATGATv2：全局节点类型映射、均值和最大值读出、训练集类别权重" \
  --output "$LOGDIR" > "$LOG" 2>&1 &

echo $!

